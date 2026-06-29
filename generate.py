import argparse
import os
from agents.local_llm import LocalLLM
from agents.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="AI-Learning-Agent: Autonomous Curriculum Generator")
    parser.add_argument("--subject", type=str, required=True, help="The subject to teach (e.g., 'Machine Learning')")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="The HuggingFace model ID to use in vLLM")
    args = parser.parse_args()

    print("========================================")
    print(f"Initializing AI-Learning-Agent")
    print(f"Target Subject: {args.subject}")
    print(f"Model: {args.model}")
    print("========================================\n")

    # Initialize local LLM
    llm = LocalLLM(model_name=args.model, dtype="bfloat16")
    
    # Initialize orchestrator
    orchestrator = Orchestrator(llm)
    
    # Define output directory
    out_dir = os.path.join("generated", args.subject.replace(" ", "_"))
    
    # Start generation
    orchestrator.generate_curriculum(args.subject, out_dir)
    
    print("\nGeneration complete! Check the 'generated/' folder.")

if __name__ == "__main__":
    main()
