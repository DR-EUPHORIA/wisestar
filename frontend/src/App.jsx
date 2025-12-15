import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatHome from './pages/ChatHome';
import Features from './pages/Features';
import Profile from './pages/Profile';
import SolveProblem from './pages/SolveProblem';
import GenerateQuestion from './pages/GenerateQuestion';
import LearningMemory from './pages/LearningMemory';
import Statistics from './pages/Statistics';
import DailyQuestion from './pages/DailyQuestion';
import ExamPaper from './pages/ExamPaper';
import TestChat from './pages/TestChat';
import GeometryPlot from './pages/GeometryPlot';

function App() {
  return (
    <div className="flex h-screen bg-gradient-to-br from-primary-50 via-white to-purple-50">
      <Sidebar />
      <main className="flex-1 ml-0 md:ml-16 lg:ml-72 overflow-hidden">
        <Routes>
          <Route path="/" element={<ChatHome />} />
          <Route path="/chat" element={<ChatHome />} />
          <Route path="/features" element={<Features />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/solve" element={<div className="h-full overflow-y-auto p-4 md:p-10"><SolveProblem /></div>} />
          <Route path="/generate" element={<div className="h-full overflow-y-auto p-4 md:p-10"><GenerateQuestion /></div>} />
          <Route path="/memory" element={<div className="h-full overflow-y-auto p-4 md:p-10"><LearningMemory /></div>} />
          <Route path="/statistics" element={<div className="h-full overflow-y-auto p-4 md:p-10"><Statistics /></div>} />
          <Route path="/daily" element={<div className="h-full overflow-y-auto p-4 md:p-10"><DailyQuestion /></div>} />
          <Route path="/exam-paper" element={<div className="h-full overflow-y-auto"><ExamPaper /></div>} />
          <Route path="/test-chat" element={<div className="h-full overflow-y-auto"><TestChat /></div>} />
          <Route path="/plot" element={<div className="h-full overflow-y-auto"><GeometryPlot /></div>} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
