import os
import re
from typing import List
from .local_llm import LocalLLM

def load_prompt(prompt_name: str) -> str:
    path = os.path.join("prompts", f"{prompt_name}.md")
    with open(path, "r") as f:
        return f.read()

class Orchestrator:
    def __init__(self, llm: LocalLLM):
        self.llm = llm
        
        # Load system prompts
        self.prompts = {
            "teacher": load_prompt("teacher"),
            "reviewer": load_prompt("reviewer"),
            "planner": load_prompt("planner"),
            "quiz_generator": load_prompt("quiz_generator"),
            "flashcard_generator": load_prompt("flashcard_generator")
        }

    def generate_curriculum(self, subject: str, out_dir: str):
        print(f"\n--- Planning Curriculum for {subject} ---")
        messages = [
            {"role": "system", "content": self.prompts["planner"]},
            {"role": "user", "content": f"Generate a detailed curriculum for: {subject}"}
        ]
        
        # We process 1 item in batch
        planner_response = self.llm.generate([messages], temperature=0.7, max_tokens=2048)[0]
        
        # Save plan
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "00_Curriculum.md"), "w") as f:
            f.write(planner_response)
            
        print(f"Curriculum planned and saved to {out_dir}/00_Curriculum.md")
        
        # Extract topics
        topics = self._extract_topics_from_plan(planner_response)
        print(f"Found {len(topics)} topics to generate.")
        
        for idx, topic in enumerate(topics):
            self._process_topic(subject, topic, idx+1, out_dir)

    def _extract_topics_from_plan(self, plan: str) -> List[str]:
        # Simple extraction: look for bullet points or numbered lists that look like topics
        topics = []
        for line in plan.split('\n'):
            line = line.strip()
            # Match "- Topic" or "1. Topic" or "## Topic"
            match = re.match(r'^(?:[-*]|\d+\.|##?)\s+(.*)', line)
            if match:
                clean_topic = match.group(1).strip()
                # Exclude obvious non-topics
                if clean_topic.lower() not in ['introduction', 'conclusion', 'summary']:
                    topics.append(clean_topic)
        return topics

    def _process_topic(self, subject: str, topic: str, index: int, base_out_dir: str):
        print(f"\n>> Generating material for Topic {index}: {topic}")
        
        # 1. Teacher Generation
        teacher_msg = [
            {"role": "system", "content": self.prompts["teacher"]},
            {"role": "user", "content": f"Teach the topic '{topic}' in the context of '{subject}'."}
        ]
        content = self.llm.generate([teacher_msg], max_tokens=8192)[0]
        
        # 2. Reviewer Optimization Loop
        max_retries = 3
        for attempt in range(max_retries):
            print(f"  [Review] Attempt {attempt+1}/{max_retries}")
            reviewer_msg = [
                {"role": "system", "content": self.prompts["reviewer"]},
                {"role": "user", "content": f"Topic: {topic}\n\nContent:\n{content}"}
            ]
            feedback = self.llm.generate([reviewer_msg], temperature=0.2, max_tokens=1024)[0]
            
            # Extract score using regex
            score_match = re.search(r'Score:\s*(\d+)/100', feedback)
            score = int(score_match.group(1)) if score_match else 0
            
            print(f"  [Review] Score: {score}/100")
            if score >= 90:
                print("  [Review] Passed!")
                break
            else:
                print(f"  [Review] Failed. Refining content...")
                # Ask teacher to refine
                teacher_msg.append({"role": "assistant", "content": content})
                teacher_msg.append({"role": "user", "content": f"The reviewer gave a score of {score}/100 and provided this feedback:\n{feedback}\n\nPlease rewrite the entire topic to address these concerns, improve the score, and ensure all MIT-level requirements are met."})
                content = self.llm.generate([teacher_msg], max_tokens=8192)[0]

        # 3. Save Topic Content
        clean_filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', topic)
        topic_filename = f"{index:02d}_{clean_filename}.md"
        topic_path = os.path.join(base_out_dir, "topics", topic_filename)
        os.makedirs(os.path.dirname(topic_path), exist_ok=True)
        with open(topic_path, "w") as f:
            f.write(content)
            
        # 4. Generate Quiz
        quiz_msg = [
            {"role": "system", "content": self.prompts["quiz_generator"]},
            {"role": "user", "content": f"Generate a quiz for this topic based on the following material:\n\n{content}"}
        ]
        quiz = self.llm.generate([quiz_msg], max_tokens=2048)[0]
        quiz_path = os.path.join(base_out_dir, "quizzes", f"Quiz_{topic_filename}")
        os.makedirs(os.path.dirname(quiz_path), exist_ok=True)
        with open(quiz_path, "w") as f:
            f.write(quiz)
            
        # 5. Generate Flashcards
        fc_msg = [
            {"role": "system", "content": self.prompts["flashcard_generator"]},
            {"role": "user", "content": f"Generate flashcards for this topic based on the following material:\n\n{content}"}
        ]
        flashcards = self.llm.generate([fc_msg], max_tokens=2048)[0]
        fc_path = os.path.join(base_out_dir, "flashcards", f"FC_{topic_filename}")
        os.makedirs(os.path.dirname(fc_path), exist_ok=True)
        with open(fc_path, "w") as f:
            f.write(flashcards)
            
        print(f"<< Finished Topic {index}")
