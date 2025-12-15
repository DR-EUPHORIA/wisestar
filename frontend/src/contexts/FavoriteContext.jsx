import React, { createContext, useContext, useState, useEffect } from 'react';

const FavoriteContext = createContext();

export const useFavoriteContext = () => {
  const context = useContext(FavoriteContext);
  if (!context) {
    throw new Error('useFavoriteContext must be used within FavoriteProvider');
  }
  return context;
};

export const FavoriteProvider = ({ children }) => {
  const [favorites, setFavorites] = useState([]);
  const [wrongQuestions, setWrongQuestions] = useState([]);
  const [tags, setTags] = useState(['重点', '易错', '考前复习', '待复习']);

  useEffect(() => {
    const savedFavorites = JSON.parse(localStorage.getItem('favorites') || '[]');
    const savedWrongQuestions = JSON.parse(localStorage.getItem('wrongQuestions') || '[]');
    const savedTags = JSON.parse(localStorage.getItem('customTags') || '[]');
    
    setFavorites(savedFavorites);
    setWrongQuestions(savedWrongQuestions);
    if (savedTags.length > 0) {
      setTags([...new Set([...tags, ...savedTags])]);
    }
  }, []);

  const saveFavorites = (newFavorites) => {
    setFavorites(newFavorites);
    localStorage.setItem('favorites', JSON.stringify(newFavorites));
  };

  const saveWrongQuestions = (newWrongQuestions) => {
    setWrongQuestions(newWrongQuestions);
    localStorage.setItem('wrongQuestions', JSON.stringify(newWrongQuestions));
  };

  const saveTags = (newTags) => {
    setTags(newTags);
    localStorage.setItem('customTags', JSON.stringify(newTags));
  };

  const addToFavorites = (question) => {
    const newItem = {
      id: Date.now(),
      question: question.question || question.content,
      answer: question.answer,
      solution: question.solution || question.steps,
      tags: [],
      note: '',
      createdAt: new Date().toISOString(),
      difficulty: question.difficulty || '中等',
      topic: question.topic || '综合'
    };
    
    const newFavorites = [newItem, ...favorites];
    saveFavorites(newFavorites);
    return newItem.id;
  };

  const addToWrongQuestions = (question) => {
    const newItem = {
      id: Date.now(),
      question: question.question || question.content,
      answer: question.answer,
      solution: question.solution || question.steps,
      userAnswer: question.userAnswer || '',
      tags: [],
      note: '',
      createdAt: new Date().toISOString(),
      retryCount: 0,
      lastRetry: null,
      difficulty: question.difficulty || '中等',
      topic: question.topic || '综合',
      mastered: false
    };
    
    const newWrongQuestions = [newItem, ...wrongQuestions];
    saveWrongQuestions(newWrongQuestions);
    return newItem.id;
  };

  const removeFromFavorites = (id) => {
    const newFavorites = favorites.filter(item => item.id !== id);
    saveFavorites(newFavorites);
  };

  const removeFromWrongQuestions = (id) => {
    const newWrongQuestions = wrongQuestions.filter(item => item.id !== id);
    saveWrongQuestions(newWrongQuestions);
  };

  const updateFavoriteNote = (id, note) => {
    const newFavorites = favorites.map(item => 
      item.id === id ? { ...item, note } : item
    );
    saveFavorites(newFavorites);
  };

  const updateWrongQuestionNote = (id, note) => {
    const newWrongQuestions = wrongQuestions.map(item => 
      item.id === id ? { ...item, note } : item
    );
    saveWrongQuestions(newWrongQuestions);
  };

  const updateFavoriteTags = (id, newTags) => {
    const newFavorites = favorites.map(item => 
      item.id === id ? { ...item, tags: newTags } : item
    );
    saveFavorites(newFavorites);
  };

  const updateWrongQuestionTags = (id, newTags) => {
    const newWrongQuestions = wrongQuestions.map(item => 
      item.id === id ? { ...item, tags: newTags } : item
    );
    saveWrongQuestions(newWrongQuestions);
  };

  const markAsMastered = (id) => {
    const newWrongQuestions = wrongQuestions.map(item => 
      item.id === id ? { ...item, mastered: true } : item
    );
    saveWrongQuestions(newWrongQuestions);
  };

  const retryWrongQuestion = (id) => {
    const newWrongQuestions = wrongQuestions.map(item => 
      item.id === id ? { 
        ...item, 
        retryCount: item.retryCount + 1,
        lastRetry: new Date().toISOString()
      } : item
    );
    saveWrongQuestions(newWrongQuestions);
  };

  const addCustomTag = (tag) => {
    if (!tags.includes(tag)) {
      const newTags = [...tags, tag];
      saveTags(newTags);
    }
  };

  const removeCustomTag = (tag) => {
    const newTags = tags.filter(t => t !== tag);
    saveTags(newTags);
  };

  const isFavorite = (questionText) => {
    return favorites.some(item => item.question === questionText);
  };

  const isWrongQuestion = (questionText) => {
    return wrongQuestions.some(item => item.question === questionText);
  };

  const value = {
    favorites,
    wrongQuestions,
    tags,
    addToFavorites,
    addToWrongQuestions,
    removeFromFavorites,
    removeFromWrongQuestions,
    updateFavoriteNote,
    updateWrongQuestionNote,
    updateFavoriteTags,
    updateWrongQuestionTags,
    markAsMastered,
    retryWrongQuestion,
    addCustomTag,
    removeCustomTag,
    isFavorite,
    isWrongQuestion
  };

  return (
    <FavoriteContext.Provider value={value}>
      {children}
    </FavoriteContext.Provider>
  );
};
