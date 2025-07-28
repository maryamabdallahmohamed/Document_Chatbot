from langchain.memory import ChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import threading
import json
import os
from datetime import datetime

class ConversationHistoryManager:
    def __init__(self, storage_path="logs/conversation_history"):
        self._conversations= {}
        self._lock = threading.Lock()
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.load_all_conversations()
    
    def get_conversation(self, conversation_id):
        """Get or create a conversation history for the given ID"""
        with self._lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = ChatMessageHistory()
                self._load_conversation_from_disk(conversation_id)
            return self._conversations[conversation_id]
    
    def add_user_message(self, conversation_id, message):
        """Add a user message to the conversation"""
        history = self.get_conversation(conversation_id)
        history.add_user_message(message)
        self._save_conversation_to_disk(conversation_id)
    
    def add_ai_message(self, conversation_id, message):
        """Add an AI message to the conversation"""
        history = self.get_conversation(conversation_id)
        history.add_ai_message(message)
        self._save_conversation_to_disk(conversation_id)
    
    def get_messages(self, conversation_id, limit):
        """Get messages for a conversation, optionally limited to recent messages"""
        history = self.get_conversation(conversation_id)
        messages = history.messages
        if limit and len(messages) > limit:
            return messages[-limit:]
        return messages
    
    def get_conversation_context(self, conversation_id, limit) :
        """Get formatted conversation context for prompt injection"""
        messages = self.get_messages(conversation_id, limit)
        if not messages:
            return ""
        
        context_lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                context_lines.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_lines.append(f"Assistant: {msg.content}")
        
        return "\n".join(context_lines)
    
    def clear_conversation(self, conversation_id):
        """Clear a specific conversation"""
        with self._lock:
            if conversation_id in self._conversations:
                self._conversations[conversation_id].clear()
                self._save_conversation_to_disk(conversation_id)
    
    def get_all_conversation_ids(self) :
        """Get all conversation IDs"""
        return list(self._conversations.keys())
    
    def delete_conversation(self, conversation_id):
        """Delete a conversation completely"""
        with self._lock:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
            
            # Remove from disk
            file_path = os.path.join(self.storage_path, f"{conversation_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def _save_conversation_to_disk(self, conversation_id):
        """Save conversation to disk for persistence"""
        try:
            history = self._conversations.get(conversation_id)
            if not history:
                return
            
            # Convert messages to serializable format
            messages_data = []
            for msg in history.messages:
                messages_data.append({
                    'type': 'human' if isinstance(msg, HumanMessage) else 'ai',
                    'content': msg.content,
                    'timestamp': datetime.now().isoformat()
                })
            
            file_path = os.path.join(self.storage_path, f"{conversation_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'conversation_id': conversation_id,
                    'messages': messages_data,
                    'last_updated': datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error saving conversation {conversation_id}: {e}")
    
    def _load_conversation_from_disk(self, conversation_id):
        """Load conversation from disk"""
        try:
            file_path = os.path.join(self.storage_path, f"{conversation_id}.json")
            if not os.path.exists(file_path):
                return
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            history = self._conversations[conversation_id]
            for msg_data in data.get('messages', []):
                if msg_data['type'] == 'human':
                    history.add_user_message(msg_data['content'])
                else:
                    history.add_ai_message(msg_data['content'])
                    
        except Exception as e:
            print(f"Error loading conversation {conversation_id}: {e}")
    
    def load_all_conversations(self):
        """Load all existing conversations from disk"""
        try:
            if not os.path.exists(self.storage_path):
                return
            
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    conversation_id = filename[:-5]  # Remove .json extension
                    self._conversations[conversation_id] = ChatMessageHistory()
                    self._load_conversation_from_disk(conversation_id)
                    
        except Exception as e:
            print(f"Error loading conversations: {e}")
    
    def get_conversation_summary(self, conversation_id):
        """Get summary info about a conversation"""
        history = self.get_conversation(conversation_id)
        messages = history.messages
        
        return {
            'conversation_id': conversation_id,
            'message_count': len(messages),
            'last_message': messages[-1].content[:100] + '...' if messages else None,
            'last_message_type': 'human' if isinstance(messages[-1], HumanMessage) else 'ai' if messages else None
        }