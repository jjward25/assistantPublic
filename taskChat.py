import os
import json
import ollama
from datetime import datetime
from typing import Dict, List, Optional

class ChatSystem:
    def __init__(self):
        self.TEXT_FILES_DIR = os.path.join(os.getcwd(), 'memory')
        self.texts: Dict[str, str] = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.tasks_data: Optional[Dict] = None  # Cache for tasks
        
        # Create memory directory if it doesn't exist
        if not os.path.exists(self.TEXT_FILES_DIR):
            os.makedirs(self.TEXT_FILES_DIR)
        
        # Initialize tasks.txt if it doesn't exist
        tasks_file = os.path.join(self.TEXT_FILES_DIR, 'tasks.txt')
        if not os.path.exists(tasks_file):
            initial_tasks = {
                "open_tasks": [],
                "completed_tasks": []
            }
            with open(tasks_file, 'w', encoding='utf-8') as f:
                json.dump(initial_tasks, f, indent=2)
        
        self.load_documents()

    def load_documents(self) -> None:
        """Load all text files including tasks into memory once during initialization"""
        for filename in os.listdir(self.TEXT_FILES_DIR):
            if filename.endswith('.txt'):
                try:
                    with open(os.path.join(self.TEXT_FILES_DIR, filename), 'r', encoding='utf-8') as file:
                        self.texts[filename] = file.read()
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")

    def get_tasks(self) -> Optional[Dict]:
        """Get tasks from cache or reload if needed"""
        if self.tasks_data is None:
            try:
                self.tasks_data = json.loads(self.texts.get('tasks.txt', '{}'))
            except json.JSONDecodeError:
                return None
        return self.tasks_data

    def save_tasks(self, tasks_data: Dict) -> bool:
        """Save tasks and update cache"""
        try:
            tasks_json = json.dumps(tasks_data, indent=2)
            # Update file, memory, and cache
            with open(os.path.join(self.TEXT_FILES_DIR, 'tasks.txt'), 'w', encoding='utf-8') as file:
                file.write(tasks_json)
            self.texts['tasks.txt'] = tasks_json
            self.tasks_data = tasks_data
            return True
        except Exception as e:
            print(f"Error saving tasks: {str(e)}")
            return False

    def add_task(self, task_name: str, project: str, description: str = "No description provided", 
                 due_date: str = "No due date", relevant_links: Optional[List[str]] = None) -> bool:
        """Add a new task with optional fields"""
        tasks_data = self.get_tasks()
        if tasks_data:
            new_task = {
                "task_name": task_name,
                "project": project,
                "create_date": datetime.now().strftime("%Y-%m-%d"),
                "due_date": due_date,
                "complete_date": None,
                "description": description,
                "relevant_links": relevant_links or []
            }
            tasks_data["open_tasks"].append(new_task)
            return self.save_tasks(tasks_data)
        return False

    def complete_task(self, task_name: str) -> bool:
        """Complete a task"""
        try:
            tasks_data = self.get_tasks()
            if not tasks_data:
                return False
            
            task_to_complete = None
            remaining_tasks = []
            
            for task in tasks_data["open_tasks"]:
                if task["task_name"].strip() == task_name.strip():
                    task_to_complete = task
                else:
                    remaining_tasks.append(task)
            
            if task_to_complete:
                task_to_complete["complete_date"] = datetime.now().strftime("%Y-%m-%d")
                tasks_data["open_tasks"] = remaining_tasks
                tasks_data["completed_tasks"].append(task_to_complete)
                return self.save_tasks(tasks_data)
            
            return False
            
        except Exception as e:
            print(f"Error in complete_task: {str(e)}")
            return False

    def delete_task(self, task_name: str) -> bool:
        """Delete a task from both open and completed tasks"""
        tasks_data = self.get_tasks()
        if not tasks_data:
            return False
        
        task_found = False
        # Remove from open tasks
        original_open_count = len(tasks_data["open_tasks"])
        tasks_data["open_tasks"] = [
            task for task in tasks_data["open_tasks"]
            if task["task_name"].strip().lower() != task_name.strip().lower()
        ]
        if len(tasks_data["open_tasks"]) < original_open_count:
            task_found = True
        
        # Remove from completed tasks
        original_completed_count = len(tasks_data["completed_tasks"])
        tasks_data["completed_tasks"] = [
            task for task in tasks_data["completed_tasks"]
            if task["task_name"].strip().lower() != task_name.strip().lower()
        ]
        if len(tasks_data["completed_tasks"]) < original_completed_count:
            task_found = True
        
        if not task_found:
            return False
            
        return self.save_tasks(tasks_data)

    def create_message(self, message: str, role: str) -> Dict[str, str]:
        return {'role': role, 'content': message}

    def process_query(self, query: str) -> str:
        """Process user query and maintain conversation history"""
        # Handle delete task command
        if query.lower().startswith("delete task:"):
            task_name = query[len("delete task:"):].strip()
            try:
                success = self.delete_task(task_name)
                if success:
                    return f"Task '{task_name}' has been deleted from tasks file."
                else:
                    return f"Error: Could not find task '{task_name}'."
                
            except Exception as e:
                return f"Error deleting task: {str(e)}"
        # Handle task management commands
        if query.lower().startswith("add task:"):
            try:
                parts = query[9:].split("|")
                if len(parts) < 2:
                    return "Error: Please provide at least Task Name and Project Name. Format: add task: Task Name | Project Name"
                
                # Use provided values or defaults
                task_name = parts[0].strip()
                project = parts[1].strip()
                description = parts[2].strip() if len(parts) > 2 else "No description provided"
                due_date = parts[3].strip() if len(parts) > 3 else "No due date"
                relevant_links = [link.strip() for link in parts[4].strip()[1:-1].split(",")] if len(parts) > 4 else []
                # "add task: New Task | Project Name | Description Text | 2025-02-15 | [https://docs.example.com/schema]"
                success = self.add_task(
                    task_name=task_name,
                    project=project,
                    description=description,
                    due_date=due_date,
                    relevant_links=relevant_links
                )
                
                if success:
                    return f"Task '{task_name}' added to project '{project}' successfully!"
                return "Error adding task: Failed to save to file"
                
            except Exception as e:
                return f"Error adding task: {str(e)}"
        
        elif query.lower().startswith("complete task:"):
            task_name = query[len("complete task:"):].strip(': ')
            success = self.complete_task(task_name)
            return f"Task '{task_name}' marked as complete!" if success else f"Error: Could not find task '{task_name}'"
        
        elif query.lower() in ["what are my open tasks?","what are all my open tasks?", "show tasks","show all tasks", "list tasks", "list open tasks"]:
            tasks_data = self.get_tasks()
            if tasks_data and tasks_data.get("open_tasks"):
                response = "Open Tasks:\n"
                for task in tasks_data["open_tasks"]:
                    response += f"\n- {task['task_name']} (Due: {task['due_date']})"
                    response += f"\n  Project: {task['project']}"
                    response += f"\n  Description: {task['description']}"
                return response
            return "No open tasks found."

        # For regular queries, use simple system message
        messages = [
            self.create_message(
                "You are a helpful assistant.",
                'system'
            )
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history[-5:])
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
    print("\nWelcome to the Task Management Chat System!")
    print("You can:")
    print("- Chat naturally with the system")
    print("- Add tasks using: add task: Task Name | Project Name")
    print("- Complete tasks using: complete task: task_name")
    print("- Type 'quit' to exit")
    print("\nWhat would you like to do?\n")
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if user_input:
                response = chat_system.process_query(user_input)
                print("\n" + response + "\n")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    start_chat()