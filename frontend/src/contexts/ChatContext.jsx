import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';

const ChatContext = createContext();

export const useChatContext = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChatContext must be used within ChatProvider');
  }
  return context;
};

export const ChatProvider = ({ children }) => {
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    const savedConversations = localStorage.getItem('wisestar_conversations');
    if (savedConversations) {
      const parsed = JSON.parse(savedConversations);
      setConversations(parsed);
      if (parsed.length > 0) {
        setCurrentConversationId(parsed[0].id);
        setMessages(parsed[0].messages || []);
      }
    } else {
      createNewConversation();
    }
  }, []);

  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem('wisestar_conversations', JSON.stringify(conversations));
    }
  }, [conversations]);

  const createNewConversation = useCallback(() => {
    const newConversation = {
      id: Date.now().toString(),
      title: '新对话',
      createdAt: new Date().toISOString(),
      messages: []
    };
    setConversations(prev => [newConversation, ...prev]);
    setCurrentConversationId(newConversation.id);
    setMessages([]);
    return newConversation.id;
  }, []);

  const switchConversation = useCallback((conversationId) => {
    const conversation = conversations.find(c => c.id === conversationId);
    if (conversation) {
      setCurrentConversationId(conversationId);
      setMessages(conversation.messages || []);
    }
  }, [conversations]);

  const deleteConversation = useCallback((conversationId) => {
    setConversations(prev => {
      const filtered = prev.filter(c => c.id !== conversationId);
      if (conversationId === currentConversationId && filtered.length > 0) {
        setCurrentConversationId(filtered[0].id);
        setMessages(filtered[0].messages || []);
      } else if (filtered.length === 0) {
        createNewConversation();
      }
      return filtered;
    });
  }, [currentConversationId, createNewConversation]);

  const addMessage = useCallback((message) => {
    const newMessage = {
      id: Date.now().toString() + Math.random(),
      timestamp: new Date().toISOString(),
      ...message
    };

    setMessages(prev => [...prev, newMessage]);
    
    setConversations(prev => prev.map(conv => {
      if (conv.id === currentConversationId) {
        const updatedMessages = [...(conv.messages || []), newMessage];
        const title = conv.title === '新对话' && updatedMessages.length === 1 && message.role === 'user'
          ? message.content.slice(0, 30) + (message.content.length > 30 ? '...' : '')
          : conv.title;
        
        return {
          ...conv,
          messages: updatedMessages,
          title,
          updatedAt: new Date().toISOString()
        };
      }
      return conv;
    }));

    return newMessage;
  }, [currentConversationId]);

  const updateMessage = useCallback((messageId, updates) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId ? { ...msg, ...updates } : msg
    ));
    
    setConversations(prev => prev.map(conv => {
      if (conv.id === currentConversationId) {
        return {
          ...conv,
          messages: conv.messages.map(msg => 
            msg.id === messageId ? { ...msg, ...updates } : msg
          )
        };
      }
      return conv;
    }));
  }, [currentConversationId]);

  const clearCurrentConversation = useCallback(() => {
    setMessages([]);
    setConversations(prev => prev.map(conv => {
      if (conv.id === currentConversationId) {
        return { ...conv, messages: [], title: '新对话' };
      }
      return conv;
    }));
  }, [currentConversationId]);

  const value = {
    conversations,
    currentConversationId,
    messages,
    isProcessing,
    setIsProcessing,
    createNewConversation,
    switchConversation,
    deleteConversation,
    addMessage,
    updateMessage,
    clearCurrentConversation
  };

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  );
};
