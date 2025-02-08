import os
import json
import ollama
from datetime import datetime
from typing import Dict, List, Optional

class ChatSystem:
    def __init__(self):
        self.BASE_MEMORY_DIR = os.path.join(os.getcwd(), 'memory')
        self.CHATS_DIR = os.path.join(self.BASE_MEMORY_DIR, 'spanishChats')
        self.conversation_history: List[Dict[str, str]] = []
        self.past_chats_context: str = ""
        
        # Create necessary directories if they don't exist
        for directory in [self.BASE_MEMORY_DIR, self.CHATS_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Load past chat histories
        self.load_past_chats()

    def load_past_chats(self) -> None:
        """Load all past chat histories into context"""
        context_parts = []
        for filename in os.listdir(self.CHATS_DIR):
            if filename.endswith('.txt'):
                try:
                    with open(os.path.join(self.CHATS_DIR, filename), 'r', encoding='utf-8') as file:
                        context_parts.append(f"\nChat History {filename}:\n{file.read()}\n---")
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
        
        self.past_chats_context = "\n".join(context_parts)

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
        return {'role': role, 'content': message}

    def process_query(self, query: str) -> str:
        """Process user query and maintain conversation history"""
        if query.lower() == "cmd quit":
            if self.save_current_chat():
                return "Chat history saved. Goodbye!"
            return "Error saving chat history. Goodbye anyway!"

        # Build messages with system context and past chats
        messages = [
            self.create_message(
                "You are a helpful Spanish (Mexican) tutor. In your responses, the first thing you do is translate the text from my message into Spanish." + 
                "Then you add your conversational response - in both English and Spanish." +  
                "So your responses should always contain (1) my message translated into Spanish, and then (2) your response, in both English and Spanish." +
                "Your response should be no more than 180 characters.",
                'system'
            )
            self.create_message(
                "Here are the most recent previous conversations for context:\n" + 
                self.past_chats_context,
                'system'
            )
        ]
        
        # Add recent conversation history
        messages.extend(self.conversation_history[-5:])  # Keep last 5 messages for context
        messages.append(self.create_message(query, 'user'))
        
        # Get response
        response = ollama.chat(model='mistral', messages=messages)
        response_content = response['message']['content']
        
        # Update conversation history
        self.conversation_history.append(self.create_message(query, 'user'))
        self.conversation_history.append(self.create_message(response_content, 'assistant'))
        
        return response_content

def start_chat():
    chat_system = ChatSystem()
    print("\nWelcome to the Spanish Chat System!")
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