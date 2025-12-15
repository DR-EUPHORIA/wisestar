import React from 'react';
import ReactDOM from 'react-dom/client';
import { HashRouter } from 'react-router-dom';
import App from './App';
import { ChatProvider } from './contexts/ChatContext';
import { FavoriteProvider } from './contexts/FavoriteContext';
import './styles/index.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <HashRouter>
      <ChatProvider>
        <FavoriteProvider>
          <App />
        </FavoriteProvider>
      </ChatProvider>
    </HashRouter>
  </React.StrictMode>
);
