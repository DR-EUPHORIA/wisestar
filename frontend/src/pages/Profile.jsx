import React, { useState, useRef } from 'react';
import { User, Mail, Calendar, Award, TrendingUp, Settings, Edit2, Save, Camera, Upload } from 'lucide-react';

const DEFAULT_AVATAR = 'https://avatars.githubusercontent.com/u/9919?s=200&v=4';

const Profile = () => {
  const [editing, setEditing] = useState(false);
  const [userInfo, setUserInfo] = useState({
    name: '数学学习者',
    email: 'user@example.com',
    joinDate: '2025-01-01',
    avatar: DEFAULT_AVATAR
  });
  const fileInputRef = useRef(null);

  const stats = [
    { label: '累计解题', value: '156', icon: Award, color: 'text-blue-600' },
    { label: '生成题目', value: '89', icon: TrendingUp, color: 'text-purple-600' },
    { label: '学习天数', value: '45', icon: Calendar, color: 'text-green-600' },
    { label: '成功率', value: '76%', icon: TrendingUp, color: 'text-orange-600' }
  ];

  const achievements = [
    { title: '初学者', description: '完成首次解题', earned: true },
    { title: '勤奋学习', description: '连续学习7天', earned: true },
    { title: '题海战术', description: '累计解题100道', earned: true },
    { title: '精益求精', description: '成功率达到80%', earned: false },
    { title: '出题达人', description: '生成题目100道', earned: false },
    { title: '学习之星', description: '连续学习30天', earned: false }
  ];

  const handleAvatarChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 5 * 1024 * 1024) {
        alert('图片大小不能超过5MB');
        return;
      }
      const reader = new FileReader();
      reader.onloadend = () => {
        setUserInfo({...userInfo, avatar: reader.result});
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSave = () => {
    localStorage.setItem('userProfile', JSON.stringify(userInfo));
    setEditing(false);
  };

  React.useEffect(() => {
    const saved = localStorage.getItem('userProfile');
    if (saved) {
      const parsed = JSON.parse(saved);
      setUserInfo({...parsed, avatar: parsed.avatar || DEFAULT_AVATAR});
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-purple-50 py-6 md:py-12 px-4 md:px-6">
      <div className="max-w-5xl mx-auto space-y-4 md:space-y-6">
        <div className="bg-white rounded-2xl shadow-medium p-4 md:p-8 border border-slate-200/60">
          <div className="flex items-start justify-between mb-6">
            <h1 className="text-2xl md:text-3xl font-bold text-slate-900">个人中心</h1>
            <button
              onClick={() => editing ? handleSave() : setEditing(true)}
              className="flex items-center gap-2 px-3 md:px-4 py-2 text-primary-600 hover:bg-primary-50 rounded-lg transition-colors text-sm md:text-base"
            >
              {editing ? (
                <>
                  <Save className="w-4 h-4" />
                  <span className="hidden sm:inline">保存</span>
                </>
              ) : (
                <>
                  <Edit2 className="w-4 h-4" />
                  <span className="hidden sm:inline">编辑</span>
                </>
              )}
            </button>
          </div>

          <div className="flex flex-col sm:flex-row items-start gap-6 md:gap-8">
            <div className="flex-shrink-0 relative group mx-auto sm:mx-0">
              <div className="w-24 h-24 md:w-28 md:h-28 rounded-2xl overflow-hidden shadow-lg ring-4 ring-white">
                {userInfo.avatar ? (
                  <img 
                    src={userInfo.avatar} 
                    alt="Avatar" 
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full bg-gradient-to-br from-primary-600 to-primary-700 flex items-center justify-center text-white text-3xl font-bold">
                    {userInfo.name.charAt(0)}
                  </div>
                )}
              </div>
              {editing && (
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="absolute inset-0 bg-black/50 rounded-2xl flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <Camera className="w-8 h-8 text-white" />
                </button>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleAvatarChange}
                className="hidden"
              />
            </div>

            <div className="flex-1 space-y-4 w-full">
              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">姓名</label>
                {editing ? (
                  <input
                    type="text"
                    value={userInfo.name}
                    onChange={(e) => setUserInfo({...userInfo, name: e.target.value})}
                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
                  />
                ) : (
                  <p className="text-lg text-slate-900">{userInfo.name}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">邮箱</label>
                {editing ? (
                  <input
                    type="email"
                    value={userInfo.email}
                    onChange={(e) => setUserInfo({...userInfo, email: e.target.value})}
                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
                  />
                ) : (
                  <p className="text-slate-600">{userInfo.email}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">加入时间</label>
                <p className="text-slate-600">{userInfo.joinDate}</p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4">
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <div key={index} className="bg-white rounded-xl p-4 md:p-6 shadow-soft border border-slate-200/60">
                <Icon className={`w-6 h-6 md:w-8 md:h-8 ${stat.color} mb-2 md:mb-3`} />
                <div className="text-2xl md:text-3xl font-bold text-slate-900 mb-1">{stat.value}</div>
                <div className="text-xs md:text-sm text-slate-600">{stat.label}</div>
              </div>
            );
          })}
        </div>

        <div className="bg-white rounded-2xl shadow-medium p-4 md:p-8 border border-slate-200/60">
          <h2 className="text-xl md:text-2xl font-bold text-slate-900 mb-4 md:mb-6">成就徽章</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 md:gap-4">
            {achievements.map((achievement, index) => (
              <div
                key={index}
                className={`p-4 rounded-xl border-2 transition-all ${
                  achievement.earned
                    ? 'bg-gradient-to-br from-primary-50 to-purple-50 border-primary-200'
                    : 'bg-slate-50 border-slate-200 opacity-60'
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                    achievement.earned ? 'bg-primary-600' : 'bg-slate-400'
                  }`}>
                    <Award className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-bold text-slate-900 mb-1 text-sm md:text-base">{achievement.title}</h3>
                    <p className="text-xs md:text-sm text-slate-600">{achievement.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-medium p-4 md:p-8 border border-slate-200/60">
          <h2 className="text-xl md:text-2xl font-bold text-slate-900 mb-4 md:mb-6 flex items-center gap-2">
            <Settings className="w-5 h-5 md:w-6 md:h-6" />
            设置
          </h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between py-3 border-b border-slate-200">
              <div className="flex-1 min-w-0 mr-4">
                <h3 className="font-semibold text-slate-900 text-sm md:text-base">通知提醒</h3>
                <p className="text-xs md:text-sm text-slate-600">接收学习提醒和每日推荐</p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer flex-shrink-0">
                <input type="checkbox" className="sr-only peer" defaultChecked />
                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
              </label>
            </div>

            <div className="flex items-center justify-between py-3 border-b border-slate-200">
              <div className="flex-1 min-w-0 mr-4">
                <h3 className="font-semibold text-slate-900 text-sm md:text-base">数据统计</h3>
                <p className="text-xs md:text-sm text-slate-600">记录学习数据用于分析</p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer flex-shrink-0">
                <input type="checkbox" className="sr-only peer" defaultChecked />
                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
              </label>
            </div>

            <div className="flex items-center justify-between py-3">
              <div className="flex-1 min-w-0 mr-4">
                <h3 className="font-semibold text-slate-900 text-sm md:text-base">自动保存</h3>
                <p className="text-xs md:text-sm text-slate-600">自动保存对话和解题记录</p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer flex-shrink-0">
                <input type="checkbox" className="sr-only peer" defaultChecked />
                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
              </label>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile;
