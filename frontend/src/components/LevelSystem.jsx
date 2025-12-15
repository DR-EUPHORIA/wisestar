import React, { useState, useEffect } from 'react';
import { Zap, TrendingUp } from 'lucide-react';

const LevelSystem = ({ compact = false }) => {
  const [userLevel, setUserLevel] = useState(1);
  const [userExp, setUserExp] = useState(0);
  const [expToNextLevel, setExpToNextLevel] = useState(100);

  useEffect(() => {
    const savedExp = parseInt(localStorage.getItem('userExp') || '0');
    setUserExp(savedExp);
    calculateLevel(savedExp);

    const interval = setInterval(() => {
      const currentExp = parseInt(localStorage.getItem('userExp') || '0');
      if (currentExp !== userExp) {
        setUserExp(currentExp);
        calculateLevel(currentExp);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const calculateLevel = (exp) => {
    let level = 1;
    let expNeeded = 100;
    let totalExp = 0;

    while (exp >= totalExp + expNeeded) {
      totalExp += expNeeded;
      level++;
      expNeeded = Math.floor(expNeeded * 1.5);
    }

    setUserLevel(level);
    setExpToNextLevel(expNeeded);
    setUserExp(exp - totalExp);
  };

  const getLevelTitle = (level) => {
    if (level < 5) return '初学者';
    if (level < 10) return '学徒';
    if (level < 20) return '熟练者';
    if (level < 30) return '专家';
    if (level < 50) return '大师';
    return '宗师';
  };

  const progress = (userExp / expToNextLevel) * 100;

  if (compact) {
    return (
      <div className="flex items-center gap-3 px-4 py-2 bg-gradient-to-r from-primary-50 to-purple-50 rounded-xl border border-primary-200">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-600 to-primary-700 flex items-center justify-center text-white font-bold text-sm">
            {userLevel}
          </div>
          <div className="flex flex-col">
            <span className="text-xs font-semibold text-slate-700">{getLevelTitle(userLevel)}</span>
            <div className="flex items-center gap-1">
              <Zap className="w-3 h-3 text-yellow-500" />
              <span className="text-xs text-slate-600">{userExp}/{expToNextLevel}</span>
            </div>
          </div>
        </div>
        <div className="flex-1 h-2 bg-slate-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-primary-500 to-purple-500 transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="card-elevated bg-gradient-to-br from-primary-50 to-purple-50 border-2 border-primary-200">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-600 to-primary-700 flex items-center justify-center text-white font-bold text-2xl shadow-lg">
            {userLevel}
          </div>
          <div>
            <h3 className="text-xl font-bold text-slate-900">{getLevelTitle(userLevel)}</h3>
            <p className="text-sm text-slate-600">等级 {userLevel}</p>
          </div>
        </div>
        <TrendingUp className="w-8 h-8 text-primary-600" />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-600">经验值</span>
          <span className="font-semibold text-slate-900">
            {userExp} / {expToNextLevel}
          </span>
        </div>
        <div className="h-4 bg-slate-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-primary-500 to-purple-500 transition-all duration-500 flex items-center justify-end pr-2"
            style={{ width: `${progress}%` }}
          >
            {progress > 20 && (
              <span className="text-xs font-bold text-white">{Math.floor(progress)}%</span>
            )}
          </div>
        </div>
        <p className="text-xs text-slate-500 text-center">
          还需 {expToNextLevel - userExp} 经验升级
        </p>
      </div>

      <div className="mt-4 pt-4 border-t border-primary-200">
        <h4 className="text-sm font-semibold text-slate-700 mb-2">获取经验值</h4>
        <div className="space-y-1 text-xs text-slate-600">
          <div className="flex items-center justify-between">
            <span>• 每日打卡</span>
            <span className="font-semibold text-primary-600">+5 经验</span>
          </div>
          <div className="flex items-center justify-between">
            <span>• 完成每日一题</span>
            <span className="font-semibold text-primary-600">+10 经验</span>
          </div>
          <div className="flex items-center justify-between">
            <span>• 智能解题</span>
            <span className="font-semibold text-primary-600">+15 经验</span>
          </div>
          <div className="flex items-center justify-between">
            <span>• 智能组卷</span>
            <span className="font-semibold text-primary-600">+20 经验</span>
          </div>
          <div className="flex items-center justify-between">
            <span>• 连续学习7天</span>
            <span className="font-semibold text-primary-600">+50 经验</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LevelSystem;
