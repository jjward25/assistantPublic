import os
import json
import ollama
from datetime import datetime
from typing import Dict, List, Optional
from functools import lru_cache  # Add this import at the top

class ChatSystem:
    def __init__(self):
        self.BASE_MEMORY_DIR = os.path.join(os.getcwd(), 'memory')
        self.CHATS_DIR = os.path.join(self.BASE_MEMORY_DIR, 'therapyChats')
        self.conversation_history: List[Dict[str, str]] = []
        self.past_chats_context: str = ""
        self.latest_chat_date: Optional[datetime] = None
        self.MAX_HISTORY = 50  # Maximum messages to keep in memory
        self.system_messages = None  # Cache for system messages
        
        # Create necessary directories if they don't exist
        for directory in [self.BASE_MEMORY_DIR, self.CHATS_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initial load of past chats
        self.load_past_chats()
        # Initialize system messages
        self.init_system_messages()

    @lru_cache(maxsize=1)
    def init_system_messages(self) -> List[Dict[str, str]]:
        """Cache system messages to avoid recreating them on every query"""
        now = datetime.now()
        days_since_last_chat = 0
        
        if self.latest_chat_date:
            time_since_last_chat = now - self.latest_chat_date
            days_since_last_chat = time_since_last_chat.days

        message1 = "You always maintain a keen awareness of the history of the conversation and the user's responses. You strive to be as productive as possible - you do not waste time in building your understanding of the user, or in planning and executing your therapy."
        
        message2 = "You are an intelligent therapist who's only goal is to improve the user's emotional wellbeing using evidence-based " + \
            "therapeutic techniques. Your goal is to identify concerns, develop strategies for coping with them, and promote " + \
            "long-lasting personal growth. During our sessions, you will employ various techniques such as cognitive behavioral therapy (CBT), " + \
            "dialectical behavior therapy (DBT), solution-focused brief therapy (SFBT), psychodynamic therapy, and mindfulness-based interventions. " + \
            "you will adapt these methods according to your unique needs and circumstances, while continuously monitoring the effectiveness of each technique in real-time. " + \
            "It is essential for our progress that we maintain a structured approach towards our sessions. This means sticking to predefined agendas and being honest with one another."

        message3 = "Here are the previous chat sessions for context:\n" + self.past_chats_context

        self.system_messages = [
            self.create_message(message1, 'system'),
            self.create_message(message2, 'system'),
            self.create_message(message3, 'system')
        ]
        return self.system_messages

    def load_past_chats(self) -> None:
        """Load all past chat histories into context and find most recent chat"""
        context_parts = []
        latest_chat_date = None
        
        # Get list of chat files sorted by date (newest first)
        chat_files = sorted(
            [f for f in os.listdir(self.CHATS_DIR) if f.endswith('.txt')],
            key=lambda x: os.path.getmtime(os.path.join(self.CHATS_DIR, x)),
            reverse=True
        )
        
        # Only load the 5 most recent chat files to avoid context overflow
        for filename in chat_files[:15]:
            try:
                date_str = filename.replace('chat_', '').replace('.txt', '')
                chat_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                
                if latest_chat_date is None or chat_date > latest_chat_date:
                    latest_chat_date = chat_date
                
                with open(os.path.join(self.CHATS_DIR, filename), 'r', encoding='utf-8') as file:
                    context_parts.append(f"\nChat History from {chat_date.strftime('%Y-%m-%d %H:%M')}:\n{file.read()}\n---")
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
        
        self.past_chats_context = "\n".join(context_parts)
        self.latest_chat_date = latest_chat_date

    def save_current_chat(self) -> bool:
        """Save the current chat history to a new file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_{timestamp}.txt"
            filepath = os.path.join(self.CHATS_DIR, filename)
            
            chat_content = []
            for message in self.conversation_history:
                role = "User" if message['role'] == 'user' else "Assistant"
                chat_content.append(f"{role}: {message['content']}\n")
            
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write("".join(chat_content))
            return True
        except Exception as e:
            print(f"Error saving chat history: {str(e)}")
            return False

    def create_message(self, message: str, role: str) -> Dict[str, str]:
        """Create a message dictionary for the chat history"""
        return {'role': role, 'content': message}

    def process_query(self, query: str) -> str:
        """Process user query and maintain conversation history"""
        # Handle delete task command
        if query.lower().startswith("delete task:"):
            task_name = query[len("delete task:"):].strip()
            try:
                # Find and remove the task from conversation history
                new_history = []
                task_found = False
                for message in self.conversation_history:
                    if task_name not in message['content']:
                        new_history.append(message)
                    else:
                        task_found = True
                
                if task_found:
                    self.conversation_history = new_history
                    return f"Task '{task_name}' has been deleted."
                else:
                    return f"Task '{task_name}' not found."
                
            except Exception as e:
                return f"Error deleting task: {str(e)}"

        # Handle quit command
        if query.lower() == "cmd quit":
            if self.save_current_chat():
                return "Chat history saved. Goodbye!"
            return "Error saving chat history. Goodbye anyway!"

        # Calculate time since last chat
        now = datetime.now()
        days_since_last_chat = 0
        if self.latest_chat_date:
            time_since_last_chat = now - self.latest_chat_date
            days_since_last_chat = time_since_last_chat.days

        # Build messages array
        messages = self.system_messages or self.init_system_messages()
        
        # Add ALL conversation history to storage, but limit context window for model
        self.conversation_history.append(self.create_message(query, 'user'))
        
        # Add recent messages for model context (last 25)
        messages.extend(self.conversation_history[-25:])
        
        # Get response
        response = ollama.chat(model='hf.co/PixelPanda1/WellMinded_Therapy_Engine-gguf', messages=messages)
        response_content = response['message']['content']
        
        # Save response to history
        self.conversation_history.append(self.create_message(response_content, 'assistant'))
        
        return response_content

def start_chat():
    chat_system = ChatSystem()
    print("\nWelcome to the Therapy Chat System!")
    print("- Chat naturally with the system")
    print("- Type 'cmd quit' to save and exit")
    print("\nWhat would you like to talk about?\n")
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if not user_input:
                continue
                
            response = chat_system.process_query(user_input)
            print("\n" + response + "\n")
            
            if user_input.lower() == "cmd quit":
                break
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    start_chat()