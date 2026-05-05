import { queryAPI } from "./api/query";
import { useState, useEffect, useRef } from 'react';
import { ChatSidebar } from './components/ChatSidebar';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { ChatWelcome } from './components/ChatWelcome';
import { ScrollArea } from './components/ui/scroll-area';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Plane, LogOut } from 'lucide-react';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: string | Date;
  chart?: any;
  data?: any[];
}

interface ChatThread {
  id: string;
  title: string;
  messages: Message[];
  lastUpdated: string | Date;
}

const STORAGE_KEY = 'gate_chat_threads_v1';

export default function App() {
  // Login state
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [currentUser, setCurrentUser] = useState('');

  // Chat state
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [pendingThread, setPendingThread] = useState<ChatThread | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [activeChat, setActiveChat] = useState<string | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // Check if already logged in on mount
  useEffect(() => {
    const savedUser = localStorage.getItem('aiport_user');
    if (savedUser) {
      setCurrentUser(savedUser);
      setIsLoggedIn(true);
    }
  }, []);

  // Computed values
  const activeThread = threads.find(t => t.id === activeChat) || null;
  const messages = activeThread ? activeThread.messages.map(m => ({
    ...m,
    timestamp: typeof m.timestamp === 'string' ? new Date(m.timestamp) : m.timestamp
  })) : [];
  const isWelcomeView = messages.length === 0;

  // API call
  const generateResponse = async (userMessage: string): Promise<{ answer: string; chart?: any; data?: any[] }> => {
    const res = await queryAPI(userMessage);
    return {
      answer: res.answer,
      chart: res.chart,
      data: res.tableData
    }; 
  };

  // Auto-scroll effect
  useEffect(() => {
    const t = setTimeout(() => {
      try {
        const root = scrollAreaRef.current as HTMLElement | null;
        if (!root) return;
        const viewport = root.querySelector('[data-slot="scroll-area-viewport"]');
        if (viewport) {
          (viewport as HTMLElement).scrollTo({ top: (viewport as HTMLElement).scrollHeight, behavior: 'smooth' });
        } else {
          bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
      } catch (err) {}
    }, 50);
    return () => clearTimeout(t);
  }, [messages]);

  // Load threads from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed: ChatThread[] = JSON.parse(raw);
        const hydrated = parsed.map(t => ({
          ...t,
          lastUpdated: t.lastUpdated ? new Date(t.lastUpdated).toString() : new Date().toString(),
          messages: (t.messages || []).map(m => ({
            ...m,
            timestamp: m.timestamp ? new Date(m.timestamp).toString() : new Date().toString()
          }))
        }));
        setThreads(hydrated);
      }
    } catch (err) {
      console.error('Failed to load chat threads', err);
    }
  }, []);

  // Save threads to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
    } catch (err) {
      console.error('Failed to save chat threads', err);
    }
  }, [threads]);

  // Login handler
  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError('');

    if ((username === 'admin' && password === 'admin123') || 
        (username === 'guest' && password === 'guest123')) {
      setCurrentUser(username);
      setIsLoggedIn(true);
      localStorage.setItem('aiport_user', username);
    } else {
      setLoginError('Invalid credentials');
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setCurrentUser('');
    localStorage.removeItem('aiport_user');
    setUsername('');
    setPassword('');
  };

  const handleSendMessage = async (content: string) => {
    let currentChatId = activeChat;
    if (!currentChatId) {
      currentChatId = Date.now().toString();
      setActiveChat(currentChatId);
    }

    const threadExists = threads.some(t => t.id === currentChatId);
    if (!threadExists) {
      const toAdd: ChatThread = pendingThread && pendingThread.id === currentChatId
        ? pendingThread
        : {
            id: currentChatId,
            title: 'New Chat',
            messages: [],
            lastUpdated: new Date().toString()
          };
      setThreads(prev => [toAdd, ...prev]);
      if (pendingThread && pendingThread.id === currentChatId) setPendingThread(null);
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      role: 'user',
      timestamp: new Date().toString()
    };

    setThreads(prev => prev.map(t => t.id === currentChatId ? {
      ...t,
      messages: [...t.messages, userMessage],
      lastUpdated: new Date().toString()
    } : t));

    setIsTyping(true);

    try {
      const response = await generateResponse(content);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.answer,
        role: 'assistant',
        timestamp: new Date().toString(),
        chart: response.chart,
        data: response.data
      };

      setThreads(prev => prev.map(t => t.id === currentChatId ? ({
        ...t,
        messages: [...t.messages, assistantMessage],
        lastUpdated: new Date().toString()
      }) : t));
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "I apologize, but I'm having trouble processing your request right now. Please try again in a moment.",
        role: 'assistant',
        timestamp: new Date().toString()
      };
      setThreads(prev => prev.map(t => t.id === currentChatId ? ({
        ...t,
        messages: [...t.messages, errorMessage],
        lastUpdated: new Date().toString()
      }) : t));
    } finally {
      setIsTyping(false);
    }
  };

  const handleNewChat = () => {
    const id = Date.now().toString();
    const newThread: ChatThread = {
      id,
      title: 'New Chat',
      messages: [],
      lastUpdated: new Date().toString()
    };
    setPendingThread(newThread);
    setActiveChat(id);
  };

  const handleSelectChat = (chatId: string) => {
    if (pendingThread && (!pendingThread.messages || pendingThread.messages.length === 0)) {
      setPendingThread(null);
    }
    setActiveChat(chatId);
  };

  const handleDeleteChat = (chatId: string) => {
    if (pendingThread && pendingThread.id === chatId) {
      setPendingThread(null);
      setActiveChat(prevActive => prevActive === chatId ? (threads.length ? threads[0].id : null) : prevActive);
      return;
    }
    setThreads(prev => {
      const remaining = prev.filter(t => t.id !== chatId);
      setActiveChat(prevActive => {
        if (prevActive === chatId) {
          return remaining.length ? remaining[0].id : null;
        }
        return prevActive;
      });
      return remaining;
    });
  };

  // Show login page if not logged in
  if (!isLoggedIn) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-white to-green-50 p-4">
        <Card className="w-full max-w-md shadow-lg">
          <CardHeader className="space-y-3 text-center pb-6">
            <div className="mx-auto bg-gradient-to-br from-blue-500 to-green-500 p-3 rounded-2xl w-fit">
              <Plane className="h-8 w-8 text-white" />
            </div>
            <CardTitle className="text-2xl font-bold">AIport</CardTitle>
            <CardDescription className="text-base">
              AI-Powered Airport Operations Intelligence
            </CardDescription>
          </CardHeader>
          
          <CardContent>
            <form onSubmit={handleLogin} className="space-y-4">
              {loginError && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
                  {loginError}
                </div>
              )}
              
              <div className="space-y-2">
                <label htmlFor="username" className="text-sm font-medium block">Username</label>
                <Input
                  id="username"
                  type="text"
                  placeholder="Enter your username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  className="h-11"
                />
              </div>
              
              <div className="space-y-2">
                <label htmlFor="password" className="text-sm font-medium block">Password</label>
                <Input
                  id="password"
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="h-11"
                />
              </div>
              
              <Button type="submit" className="w-full h-11 text-base font-medium">
                Sign In
              </Button>
            </form>

            <div className="mt-6 pt-6 border-t border-gray-200">
              <p className="text-xs text-center text-gray-500">
                Demo Credentials:<br />
                <span className="font-mono font-medium">admin / admin123</span>
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Main app view (after login)
  return (
    <div className="flex h-screen overflow-hidden bg-white">
      <ChatSidebar
        isCollapsed={isSidebarCollapsed}
        onToggle={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
        onNewChat={handleNewChat}
        activeChat={activeChat}
        onSelectChat={handleSelectChat}
        chatHistory={threads.map(t => ({
          id: t.id,
          title: t.title,
          timestamp: t.lastUpdated,
          preview: t.messages.length && t.messages[t.messages.length - 1]?.content 
            ? t.messages[t.messages.length - 1].content.slice(0, 80) 
            : 'No messages yet'
        }))}
        onDeleteChat={handleDeleteChat}
      />

      <div className="flex-1 flex flex-col min-w-0 min-h-0">
        {/* Header with Logout */}
        <div className="border-b border-gray-200 bg-white px-6 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-gray-900">AIport Assistant</h1>
            <p className="text-xs text-gray-500">Logged in as {currentUser}</p>
          </div>
          <Button variant="outline" size="sm" onClick={handleLogout} className="h-8 px-3">
            <LogOut className="h-4 w-4 mr-2" />
            Logout
          </Button>
        </div>

        {isWelcomeView ? (
          <ChatWelcome />
        ) : (
          <>
            <div className="flex-1 overflow-hidden">
              <ScrollArea ref={scrollAreaRef} className="h-full">
                <div className="max-w-4xl mx-auto">
                  {messages.map((message) => (
                    <ChatMessage key={message.id} message={message} />
                  ))}
                  <div ref={bottomRef} />
                  
                  {isTyping && (
                    <div className="flex gap-4 p-6 bg-gray-50/30">
                      <div className="flex-shrink-0">
                        <div className="h-8 w-8 rounded-full bg-green-100 flex items-center justify-center">
                          <span className="text-sm font-medium text-green-700">AI</span>
                        </div>
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-sm font-medium text-gray-900">Gate Assistant</span>
                          <span className="text-xs text-gray-500">typing...</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </div>
          </>
        )}

        <ChatInput onSendMessage={handleSendMessage} disabled={false} isTyping={isTyping} />
      </div>
    </div>
  );
}
