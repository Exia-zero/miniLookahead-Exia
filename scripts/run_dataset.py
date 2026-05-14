import json
import csv
import time
import uuid
from openai import OpenAI
from tqdm import tqdm
import re
import argparse
from dynasor.core.evaluator import (
    extract_answer,
    strip_string,
    math_equal,
    extract_first_boxed_answer,
)
import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
import threading
from sentence_transformers import SentenceTransformer


def load_questions(file_path):
    """Load questions from jsonl file"""
    questions = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            questions.append(data)
    return questions


def get_model_response(
    prompt,
    model="gpt-4-turbo-preview",
    temperature=0.0,
    max_tokens=1000,
    stop=None,
    client=None,
    method="baseline",
):
    """Get response from OpenAI API"""
    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            top_p=0.95,
        )
        finish_reason = response.choices[0].finish_reason
        stop_reason = get_completion_stop_reason(response.choices[0])
        # Get the response content
        completion_text = response.choices[0].text
        return completion_text, finish_reason, stop_reason
    except Exception as e:
        print(f"Error getting response: {e} {model}")
        return None


def get_completion_stop_reason(choice):
    """Get stop reason from OpenAI-compatible completion choice"""
    if hasattr(choice, "stop_reason"):
        return choice.stop_reason
    if hasattr(choice, "matched_stop"):
        return choice.matched_stop
    return None


def get_batched_model_responses(
    prompts,
    model=None,
    temperature=0.0,
    max_tokens=1000,
    stop=None,
    client=None,
):
    try:
        response = client.completions.create(
            model=model,
            prompt=prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            top_p=0.95,
        )
        ordered_choices = [None] * len(prompts)
        for fallback_idx, choice in enumerate(response.choices):
            choice_idx = getattr(choice, "index", fallback_idx)
            if choice_idx is None or choice_idx >= len(prompts):
                choice_idx = fallback_idx
            ordered_choices[choice_idx] = choice

        results = []

        for choice in ordered_choices:
            if choice is None:
                results.append((None, None, None))
            else:
                results.append(
                    (
                        choice.text,
                        choice.finish_reason,
                        get_completion_stop_reason(choice),
                    )
                )
        return results

    except Exception as e:
        print(f"Error getting batched responses: {e} {model}")
        return None


def get_model_response_chat(
    prompt,
    model="gpt-4-turbo-preview",
    temperature=0.0,
    max_tokens=1000,
    stop=None,
    client=None,
):
    """Get response from OpenAI API"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            top_p=0.95,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting response: {e} {model}")
        return None


def save_results(questions, responses, output_file):
    """Save results to CSV file"""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Response"])
        for q, r in zip(questions, responses):
            writer.writerow([q["question"], r])


equal_prompts = [
    """<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nEvaluate whether the following two reasoning steps (s1 and s2) convey exactly the same meaning. Focus on semantic similarity rather than exact wording. 

Compare the main ideas, key points, overall message, logical structure, and numerical calculations/results of both reasoning steps.

If the reasoning steps convey essentially the same meaning and generate same calculation results, respond with [aligned].
If the reasoning steps express different meanings, respond with [unaligned]. If it is too hard to determine, respond with [unaligned]

Please directly provide the final result in [aligned] or [unaligned].

Reasoning step 1 (s1):
<start_s1>
{}
<end_s1>

Reasoning step 2 (s2):
<start_s2>
{}
<end_s2><|im_end|>\n<|im_start|>assistant\n["""
]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="llm-j",
        choices=["llm-j", "emb", "baseline"],
        help="Method to run: llm-j (LLM Judge), emb (Embedding), baseline",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        help="Model path",
    )
    parser.add_argument(
        "--draft_model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Draft model path",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Judge model path",
    )
    parser.add_argument("--target_port", type=int, default=12347, help="Target server port")
    parser.add_argument("--draft_port", type=int, default=12345, help="Draft server port")
    parser.add_argument("--judge_port", type=int, default=8000, help="Judge server port")
    parser.add_argument(
        "--embedding_device",
        type=str,
        default="cuda:0",
        help="Device for embedding verifier",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding verifier model path",
    )
    parser.add_argument(
        "--dataset", type=str, default="../data/aime-2024.jsonl", help="Dataset path"
    )
    parser.add_argument("--prefix", type=str, default="judge", help="Prefix")
    parser.add_argument("--start_qid", type=int, default=None, help="Start question id")
    parser.add_argument("--end_qid", type=int, default=None, help="End question id")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Prompt index")
    parser.add_argument("--threshold", type=float, default=0.95, help="Prompt index")
    parser.add_argument("--allow_no_stop", action="store_true", help="Allow no stop")
    parser.add_argument("--max_depth", type=int, default=4, help="Lookahead depth")
    parser.add_argument(
        "--step_max_tokens", type=int, default=100, help="Max tokens per reasoning step"
    )
    parser.add_argument("--max_workers", type=int, default=1, help="Max workers")
    parser.add_argument("--max_samples", type=int, default=1, help="Max samples")
    parser.add_argument(
        "--warmup_requests",
        type=int,
        default=0,
        help="Number of target model warmup requests before timed dataset processing",
    )
    parser.add_argument(
        "--warmup_max_tokens",
        type=int,
        default=256,
        help="Max tokens for each warmup request",
    )
    return parser.parse_args()


def initialize_clients(args):
    """Initialize OpenAI clients for target, draft, and judge models"""
    target_client = [
        OpenAI(
            base_url=f"http://127.0.0.1:{args.target_port}/v1",
            api_key="None",
            timeout=100000,
        )
    ]

    draft_client = None
    judge_client = None
    if args.method in ("llm-j", "emb"):
        draft_client = [
            OpenAI(
                base_url=f"http://127.0.0.1:{args.draft_port}/v1",
                api_key="None",
                timeout=100000,
            )
        ]
        judge_client = [
            OpenAI(
                base_url=f"http://127.0.0.1:{args.judge_port}/v1",
                api_key="None",
                timeout=100000,
            )
        ]

    return target_client, draft_client, judge_client


def initialize_tokenizer(args):
    """Initialize tokenizer for token counting"""
    return AutoTokenizer.from_pretrained(args.model)


def build_problem_prompt(question_text, tokenizer):
    """Build the problem prompt using the model's chat template."""
    prompt = (
        question_text
        + "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def warmup_target_model(args, target_client, tokenizer):
    """Warm up the target vLLM server before timed dataset processing."""
    if args.warmup_requests <= 0:
        return

    prompt = build_problem_prompt("What is 123 + 456?", tokenizer)

    print(f"Warming up target model for {args.warmup_requests} requests...")
    for request_idx in range(args.warmup_requests):
        result = get_model_response(
            prompt,
            temperature=0.6,
            max_tokens=args.warmup_max_tokens,
            client=target_client[0],
            model=args.model,
        )
        if result is None:
            raise RuntimeError(
                f"Warmup request {request_idx + 1}/{args.warmup_requests} failed"
            )
        print(f"Warmup {request_idx + 1}/{args.warmup_requests} done")


def setup_output_directory(prefix):
    """Create output directory with timestamp"""
    run_prefix = prefix + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = run_prefix
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def setup_embedding_model(args):
    """Setup embedding model and similarity function for 'emb' method"""
    embedding_device = args.embedding_device
    print(f"Loading embedding model {args.embedding_model} to {embedding_device}")

    try:
        embedding_model = SentenceTransformer(
            args.embedding_model, device=embedding_device
        )
        print(f"Successfully loaded embedding model to {embedding_device}")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        embedding_model = None

    embedding_lock = threading.Lock()

    def compute_similarity(sentence1, sentence2):
        """Compute cosine similarity between two sentences"""
        if embedding_model is None:
            print("Warning: Embedding model not loaded, returning 0 similarity")
            return 0.0

        with embedding_lock:
            try:
                embeddings = embedding_model.encode([sentence1, sentence2])
                similarity = embedding_model.similarity(embeddings[0], embeddings[1])
                return similarity
            except Exception as e:
                print(f"Error computing similarity: {e}")
                return 0.0

    return compute_similarity


def process_questions_parallel(
    questions,
    args,
    target_client,
    draft_client,
    judge_client,
    tokenizer,
    output_dir,
    compute_similarity=None,
    equal_prompt=None,
):
    """Process questions in parallel using ThreadPoolExecutor"""
    temperature = 0.6
    results = []

    def process_question(q, sample_idx, question_idx):
        if "question" not in q:
            q["question"] = q["problem"]
        if "id" not in q:
            q["id"] = question_idx

        inp = build_problem_prompt(q["question"], tokenizer)
        target_prompt = inp
        t0 = time.time()
        next_sentence = ""
        generations = []
        generation_target = []
        generation_draft = []
        accepts = []
        equals = []
        appended_in_branch = False

        def append_stop_text(text, finish_reason, stop_reason):
            if text is None:
                text = ""
            if finish_reason == "stop" and stop_reason == "\n\n":
                return text + "\n\n"
            return text

        def generate_step(prompt, client, model):
            return get_model_response(
                prompt,
                temperature=temperature,
                max_tokens=args.step_max_tokens,
                stop=["\n\n"],
                client=client,
                model=model,
                method=args.method,
            )

        def generate_draft_steps(base_prompt):
            draft_steps = []
            draft_prefix = ""
            for _ in range(args.max_depth):
                draft_result = generate_step(
                    base_prompt + draft_prefix,
                    draft_client[0],
                    args.draft_model,
                )
                if draft_result is None:
                    break

                sentence_draft, finish_reason_draft, stop_reason_draft = draft_result
                draft_step_text = append_stop_text(
                    sentence_draft,
                    finish_reason_draft,
                    stop_reason_draft,
                )
                draft_steps.append(
                    {
                        "text": sentence_draft,
                        "finish_reason": finish_reason_draft,
                        "stop_reason": stop_reason_draft,
                        "step_text": draft_step_text,
                    }
                )
                draft_prefix += draft_step_text

                if finish_reason_draft == "stop" and stop_reason_draft != "\n\n":
                    break

            return draft_steps

        def build_target_prompts(base_prompt, draft_steps):
            target_prompts = []
            draft_prefix = ""
            for draft_step in draft_steps:
                target_prompts.append(base_prompt + draft_prefix)
                draft_prefix += draft_step["step_text"]
            return target_prompts

        def generate_target_steps(target_prompts):
            return get_batched_model_responses(
                target_prompts,
                temperature=temperature,
                max_tokens=args.step_max_tokens,
                stop=["\n\n"],
                client=target_client[0],
                model=args.model,
            )

        def verify_step(
            sentence_target,
            sentence_draft,
            finish_reason,
            stop_reason,
            finish_reason_draft,
        ):
            equal = None
            is_aligned = False

            if args.method == "llm-j":
                if args.prompt_idx == -1:
                    equal, _, _ = get_model_response(
                        equal_prompt.format(
                            (sentence_target or "").strip(),
                            (sentence_draft or "").strip(),
                        ),
                        temperature=0.0,
                        max_tokens=1000,
                        client=judge_client[0],
                        model=args.judge_model,
                    )
                    # print("xEqual: ", equal)
                    is_aligned = (
                        "[aligned]" in (equal or "")
                        and "[unaligned]" not in (equal or "")
                        and finish_reason == "stop"
                        and finish_reason_draft == "stop"
                    )
                else:
                    equal, _, _ = get_model_response(
                        equal_prompt.format(
                            (sentence_target or "").strip(),
                            (sentence_draft or "").strip(),
                        ),
                        temperature=0.0,
                        max_tokens=1,
                        client=judge_client[0],
                        model=args.judge_model,
                    )
                    is_aligned = (
                        "ali" in (equal or "")
                        and "un" not in (equal or "")
                        and (
                            args.allow_no_stop
                            or (finish_reason == "stop" and stop_reason == "\n\n")
                        )
                    )

            elif args.method == "emb":
                similarity = compute_similarity(
                    (sentence_target or "").strip(),
                    (sentence_draft or "").strip(),
                )
                equal = (
                    similarity.item()
                    if hasattr(similarity, "item")
                    else float(similarity)
                )
                # print("Equal: ", equal)
                is_aligned = equal > args.threshold and (
                    args.allow_no_stop
                    or (finish_reason == "stop" and stop_reason == "\n\n")
                )

            return equal, is_aligned

        def choose_step_text(
            is_aligned,
            sentence_target,
            sentence_draft,
            finish_reason,
            stop_reason,
            finish_reason_draft,
            stop_reason_draft,
        ):
            if is_aligned:
                return append_stop_text(
                    sentence_draft,
                    finish_reason_draft,
                    stop_reason_draft,
                )
            return append_stop_text(sentence_target, finish_reason, stop_reason)

        if args.method == "llm-j" or args.method == "emb":
            infos = []

            token_length = 0
            appended_in_branch = True
            while True:
                should_finish = False
                draft_steps = generate_draft_steps(inp)
                if not draft_steps:
                    break

                target_prompts = build_target_prompts(inp, draft_steps)
                target_steps = generate_target_steps(target_prompts)
                if not target_steps:
                    break

                next_sentence = ""
                for draft_step, target_step in zip(draft_steps, target_steps):
                    sentence_target, finish_reason, stop_reason = target_step
                    sentence_draft = draft_step["text"]
                    finish_reason_draft = draft_step["finish_reason"]
                    stop_reason_draft = draft_step["stop_reason"]

                    generation_target.append(sentence_target)
                    generation_draft.append(sentence_draft)

                    if finish_reason == "stop" and stop_reason != "\n\n":
                        step_text = sentence_target or ""
                        next_sentence += step_text
                        generations.append(step_text)
                        should_finish = True
                        break

                    equal, is_aligned = verify_step(
                        sentence_target,
                        sentence_draft,
                        finish_reason,
                        stop_reason,
                        finish_reason_draft,
                    )
                    step_text = choose_step_text(
                        is_aligned,
                        sentence_target,
                        sentence_draft,
                        finish_reason,
                        stop_reason,
                        finish_reason_draft,
                        stop_reason_draft,
                    )
                    accepts.append(1 if is_aligned else 0)

                    next_sentence += step_text
                    generations.append(step_text)
                    infos.append((equal, sentence_target, sentence_draft))
                    equals.append(equal)

                    if not is_aligned:
                        break

                if not next_sentence:
                    break

                inp += next_sentence
                tokens = tokenizer.encode(next_sentence, add_special_tokens=False)
                token_length += len(tokens)

                if should_finish:
                    break

                if token_length > 32768:
                    print(
                        f"Token length ({token_length}) exceeds 16000, truncating input"
                    )
                    break

        elif args.method == "baseline":
            next_sentence, finish_reason, stop_reason = get_model_response(
                inp,
                temperature=temperature,
                max_tokens=32000,
                client=target_client[0],
                model=args.model,
            )

        if not appended_in_branch:
            inp = inp + next_sentence
        t1 = time.time()
        generation_text = inp[len(target_prompt) :]
        generation_tokens = tokenizer.encode(generation_text, add_special_tokens=False)
        full_tokens = tokenizer.encode(inp, add_special_tokens=False)
        time_taken = t1 - t0
        speed = len(generation_tokens) / time_taken if time_taken > 0 else 0
        print("Done ", question_idx)

        # Save final input and answer to JSON file
        if args.method == "llm-j" or args.method == "emb":
            output_data = {
                "question_id": q["id"],
                "question": q["question"],
                "target_prompt": target_prompt,
                "generation_text": generation_text,
                "generation_tokens": generation_tokens,
                "full_text": inp,
                "full_tokens": full_tokens,
                "final_input": inp,
                "answer": inp,
                "accepts": accepts,
                "equals": equals,
                "num_accepts": sum(accepts),
                "num_accept_decisions": len(accepts),
                "accept_rate": sum(accepts) / len(accepts) if accepts else 0,
                "generations_target": generation_target,
                "generations_draft": generation_draft,
                "generations": generations,
                "gold": q["answer"],
                "infos": infos,
                "time_taken": time_taken,
                "speed": speed,
                "method": args.method,
            }
        elif args.method == "baseline":
            output_data = {
                "question_id": q["id"],
                "question": q["question"],
                "target_prompt": target_prompt,
                "generation_text": generation_text,
                "generation_tokens": generation_tokens,
                "full_text": inp,
                "full_tokens": full_tokens,
                "final_input": inp,
                "answer": inp,
                "accepts": accepts,
                "equals": equals,
                "num_accepts": sum(accepts),
                "num_accept_decisions": len(accepts),
                "accept_rate": sum(accepts) / len(accepts) if accepts else 0,
                "generations_target": generation_target,
                "generations_draft": generation_draft,
                "generations": generations,
                "gold": q["answer"],
                "time_taken": time_taken,
                "speed": speed,
                "method": args.method,
            }
        with open(
            output_dir + "/" + str(q["id"]) + "_" + str(sample_idx) + ".json", "w"
        ) as f:
            json.dump(output_data, f)
        print(
            "Question: ",
            q["id"],
            "answer:",
            extract_answer(inp, "aime"),
            "gold:",
            q["answer"],
            "Spec: ",
            inp is None,
        )
        return {"answer": extract_answer(inp, "aime"), "gold": q["answer"]}

    # Process questions in parallel
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for sample_idx in range(args.max_samples):
            print("Running question", sample_idx, len(questions))
            future_to_question = {
                executor.submit(process_question, q, sample_idx, question_idx): q
                for question_idx, q in enumerate(questions)
            }

        for future in tqdm(
            as_completed(future_to_question), total=len(questions) * args.max_samples
        ):
            question = future_to_question[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Question processing failed: {e}")

    return results


def calculate_and_save_accuracy(results, questions, output_dir):
    """Calculate accuracy and save results"""
    correct = 0
    for result in results:
        print(result["answer"], result["gold"])
        if math_equal(str(result["answer"]), str(result["gold"])):
            correct += 1

    accuracy = correct / len(results) if results else 0
    print(f"Accuracy: {accuracy}")

    accuracy_data = {"accuracy": correct / len(questions), "results": len(results)}
    with open(output_dir + "/" + "accuracy.json", "w") as f:
        json.dump(accuracy_data, f, indent=2)

    # Save results to CSV file
    with open(output_dir + "/" + "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["answer", "gold"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def main():
    """Main function to run dataset processing"""
    args = parse_arguments()

    # Initialize clients
    target_client, draft_client, judge_client = initialize_clients(args)

    # Initialize tokenizer
    tokenizer = initialize_tokenizer(args)

    # Setup output directory
    output_dir = setup_output_directory(args.prefix)

    # Setup embedding model if needed
    compute_similarity = None
    if args.method == "emb":
        compute_similarity = setup_embedding_model(args)

    # Load questions
    questions = load_questions(args.dataset)[args.start_qid : args.end_qid]

    # Get equal prompt if needed
    equal_prompt = None
    if args.method == "llm-j":
        equal_prompt = equal_prompts[args.prompt_idx]

    # Warmup happens outside per-question timing.
    warmup_target_model(args, target_client, tokenizer)

    # Process questions in parallel
    results = process_questions_parallel(
        questions,
        args,
        target_client,
        draft_client,
        judge_client,
        tokenizer,
        output_dir,
        compute_similarity,
        equal_prompt,
    )

    # Calculate and save accuracy
    calculate_and_save_accuracy(results, questions, output_dir)


if __name__ == "__main__":
    main()
