import os
import re
import random
import nltk
import torch
from nltk.tokenize import sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List, Dict, Any, Optional

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

class QuestionGenerator:
    """Pipeline for generating high-quality questions from text sources."""
    
    def __init__(self, model_name: str = "mrm8488/t5-base-finetuned-question-generation-ap"):
        """Initialize the question generator with a T5 model.
        
        Args:
            model_name: HuggingFace model name/path for question generation
        """
        print("Initializing Question Generator...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def clean_text(self, text: str) -> str:
        """Clean input text by removing extra whitespace, etc."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML tags if any
        text = re.sub(r'<.*?>', '', text)
        return text.strip()
    
    def segment_text(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into manageable segments for processing.
        
        Args:
            text: The input text to segment
            max_length: Maximum character length for each segment
            
        Returns:
            List of text segments
        """
        # First try to segment by sentences
        sentences = sent_tokenize(text)
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) <= max_length:
                current_segment += " " + sentence
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence
        
        if current_segment:
            segments.append(current_segment.strip())
            
        # If segments are still too large, do a hard split
        final_segments = []
        for segment in segments:
            if len(segment) > max_length:
                # Split by max_length, but try to not break words
                for i in range(0, len(segment), max_length):
                    if i + max_length < len(segment):
                        # Find the last space within the limit
                        split_point = segment[i:i+max_length].rfind(' ')
                        if split_point == -1:  # No space found, do a hard split
                            split_point = max_length
                        final_segments.append(segment[i:i+split_point].strip())
                    else:
                        final_segments.append(segment[i:].strip())
            else:
                final_segments.append(segment)
                
        return final_segments
    
    def generate_questions(self, text: str, num_questions: int = 5, 
                          temperature: float = 1.0, do_sample: bool = True) -> List[str]:
        """Generate questions from a text segment.
        
        Args:
            text: Input text segment
            num_questions: Number of questions to generate per segment
            temperature: Controls randomness (higher = more random)
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            List of generated questions
        """
        input_text = "generate questions: " + text
        
        # Encode the input and generate questions
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        # Check if input is too long and truncate if necessary
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            print(f"Warning: Input too long ({input_ids.shape[1]} tokens), truncating to {self.tokenizer.model_max_length}")
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
        
        outputs = self.model.generate(
            input_ids,
            max_length=64,
            num_return_sequences=num_questions,
            do_sample=do_sample,
            temperature=temperature,
            top_k=50,
            top_p=0.95
        )
        
        questions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return questions
    
    def is_valid_question(self, question: str) -> bool:
        """Check if a question meets basic quality criteria.
        
        Args:
            question: The question to validate
            
        Returns:
            Boolean indicating if the question is valid
        """
        # Must end with '?'
        if not question.endswith('?'):
            return False
        
        # Must have a minimum length
        if len(question.split()) < 4:
            return False
        
        # Should contain at least one question word
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'whose', 'whom', 'can', 'could', 'would', 'will', 'is', 'are', 'do', 'does', 'did']
        has_question_word = any(word.lower() in question.lower().split() for word in question_words)
        
        return has_question_word
    
    def remove_duplicates(self, questions: List[str]) -> List[str]:
        """Remove duplicate or nearly identical questions.
        
        Args:
            questions: List of questions to deduplicate
            
        Returns:
            Deduplicated list of questions
        """
        unique_questions = []
        lowercase_questions = []
        
        for question in questions:
            question_lower = question.lower()
            # Check if this question is very similar to one we've already kept
            if not any(self._similarity_score(question_lower, q) > 0.8 for q in lowercase_questions):
                unique_questions.append(question)
                lowercase_questions.append(question_lower)
                
        return unique_questions
    
    def _similarity_score(self, str1: str, str2: str) -> float:
        """Calculate a simple similarity score between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0-1)
        """
        # This is a very basic similarity check (shared words / total words)
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0
            
        shared = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return shared / total if total > 0 else 0
    
    def process_file(self, file_path: str, max_questions_per_segment: int = 10) -> List[str]:
        """Process a single text file and generate questions.
        
        Args:
            file_path: Path to the text file
            max_questions_per_segment: Maximum questions to generate per text segment
            
        Returns:
            List of generated questions
        """
        print(f"Processing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                return []
                
        # Clean the text
        text = self.clean_text(text)
        
        # Segment the text
        segments = self.segment_text(text)
        print(f"Split into {len(segments)} segments")
        
        # Generate questions for each segment
        all_questions = []
        for i, segment in enumerate(segments):
            print(f"Generating questions for segment {i+1}/{len(segments)}")
            questions = self.generate_questions(segment, num_questions=max_questions_per_segment)
            all_questions.extend(questions)
            
        return all_questions
    
    def process_directory(self, directory_path: str, max_questions_per_segment: int = 5) -> List[str]:
        """Process all text files in a directory and generate questions.
        
        Args:
            directory_path: Path to directory containing text files
            max_questions_per_segment: Maximum questions to generate per text segment
            
        Returns:
            List of generated questions
        """
        all_questions = []
        
        for filename in os.listdir(directory_path):
            if filename.endswith(('.txt', '.md')):
                file_path = os.path.join(directory_path, filename)
                questions = self.process_file(file_path, max_questions_per_segment)
                all_questions.extend(questions)
                
        return all_questions
    
    def filter_questions(self, questions: List[str]) -> List[str]:
        """Filter questions for quality.
        
        Args:
            questions: List of questions to filter
            
        Returns:
            Filtered list of high-quality questions
        """
        # Remove duplicates
        unique_questions = self.remove_duplicates(questions)
        
        # Apply basic quality filters
        filtered_questions = [q for q in unique_questions if self.is_valid_question(q)]
        
        print(f"Filtered {len(questions)} questions to {len(filtered_questions)} high-quality questions")
        return filtered_questions
    
    def select_random_questions(self, questions: List[str], n: int = 100) -> List[str]:
        """Select a random sample of questions.
        
        Args:
            questions: List of questions to sample from
            n: Number of questions to select
            
        Returns:
            Randomly selected questions
        """
        if len(questions) <= n:
            return questions
            
        return random.sample(questions, n)

    def generate_from_text(self, text: str, num_questions: int = 100) -> List[str]:
        """Generate questions directly from a text string.
        
        Args:
            text: Input text
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        # Clean the text
        text = self.clean_text(text)
        
        # Segment the text
        segments = self.segment_text(text)
        print(f"Split text into {len(segments)} segments")
        
        # Generate questions for each segment
        all_questions = []
        questions_per_segment = max(5, min(10, num_questions // len(segments) + 1))
        
        for i, segment in enumerate(segments):
            print(f"Generating questions for segment {i+1}/{len(segments)}")
            questions = self.generate_questions(segment, num_questions=questions_per_segment)
            all_questions.extend(questions)
            
        # Filter and select questions
        filtered_questions = self.filter_questions(all_questions)
        final_questions = self.select_random_questions(filtered_questions, num_questions)
        
        return final_questions


# Usage examples
if __name__ == "__main__":
    # Initialize the question generator
    qg = QuestionGenerator()
    
    # Example 1: Generate questions from a text string
    sample_text = """
    Machine learning is a subfield of artificial intelligence that focuses on developing systems that can learn from data.
    It enables computers to improve at tasks with experience. There are several types of machine learning, including
    supervised learning, unsupervised learning, and reinforcement learning. In supervised learning, algorithms learn from
    labeled training data, and make predictions based on that data. In unsupervised learning, algorithms identify
    patterns in data without labels. Reinforcement learning involves training agents to make decisions by rewarding
    desired behaviors and punishing undesired ones.
    """
    
    questions = qg.generate_from_text(sample_text, num_questions=10)
    print("\nGenerated questions from sample text:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    # Example 2: Process a specific file
    # Uncomment to use
    """
    file_questions = qg.process_file("path/to/your/file.txt")
    filtered_file_questions = qg.filter_questions(file_questions)
    final_file_questions = qg.select_random_questions(filtered_file_questions, 100)
    
    print("\nGenerated questions from file:")
    for i, q in enumerate(final_file_questions, 1):
        print(f"{i}. {q}")
    """
    
    # Example 3: Process an entire directory
    # Uncomment to use
    """
    dir_questions = qg.process_directory("path/to/your/directory")
    filtered_dir_questions = qg.filter_questions(dir_questions)
    final_dir_questions = qg.select_random_questions(filtered_dir_questions, 100)
    
    print("\nGenerated 100 random questions from directory:")
    for i, q in enumerate(final_dir_questions, 1):
        print(f"{i}. {q}")
    """