import React from 'react';
import ChatInterface from '../components/chat/ChatInterface';

const ChatHome = () => {
  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-primary-50 via-white to-purple-50">
      <ChatInterface />
    </div>
  );
};

export default ChatHome;
