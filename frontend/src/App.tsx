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
  metadata?: {
    outputFormat?: string;
    outputConfidence?: number;
    queryType?: string;
    rowCount?: number;
    sqlSource?: string;
  };
}

interface ChatThread {
  id: string;
  title: string;
  messages: Message[];
  lastUpdated: string | Date;
}

const STORAGE_KEY = 'aiport_chat_threads_v7_0_fixed';

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
  const [activeChat, setActiveChat] = useState<string | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Check if already logged in
  useEffect(() => {
    const savedUser = localStorage.getItem('aiport_user');
    if (savedUser) {
      setCurrentUser(savedUser);
      setIsLoggedIn(true);
    }
  }, []);

  // Load threads (only after login)
  useEffect(() => {
    if (!isLoggedIn) return;

    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        console.log('💾 Loaded threads:', parsed.length);
        setThreads(parsed);
        if (parsed.length > 0) {
          setActiveChat(parsed[0].id);
        }
      } catch (e) {
        console.error('Error loading threads:', e);
      }
    } else {
      const initialThread: ChatThread = {
        id: Date.now().toString(),
        title: 'New Chat',
        messages: [],
        lastUpdated: new Date().toString()
      };
      console.log('🆕 Creating initial thread:', initialThread.id);
      setThreads([initialThread]);
      setActiveChat(initialThread.id);
    }
  }, [isLoggedIn]);

  // Save threads
  useEffect(() => {
    if (threads.length > 0 && isLoggedIn) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
      console.log('💾 Saved threads:', threads.length);
    }
  }, [threads, isLoggedIn]);

  // Auto-scroll
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
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

  // Show login page
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
          </CardContent>
        </Card>
      </div>
    );
  }

  // Get current messages
  const currentThread = threads.find(t => t.id === activeChat);
  const messages = currentThread?.messages || [];

  console.log('🎨 Render - Active chat:', activeChat, 'Messages:', messages.length);

  const handleSendMessage = async (userMessage: string) => {
    console.log('\n=== HANDLE SEND MESSAGE ===');
    console.log('📤 Input:', userMessage);
    console.log('📍 Active chat:', activeChat);

    if (!activeChat) {
      console.error('❌ No active chat!');
      return;
    }

    const userMsg: Message = {
      id: Date.now().toString(),
      content: userMessage,
      role: 'user',
      timestamp: new Date().toString()
    };

    console.log('👤 User message:', userMsg.id);

    setThreads(prevThreads => {
      const updated = prevThreads.map(t => 
        t.id === activeChat 
          ? { ...t, messages: [...t.messages, userMsg], lastUpdated: new Date().toString() }
          : t
      );
      console.log('✅ User message added');
      return updated;
    });

    const currentThread = threads.find(t => t.id === activeChat);
    if (currentThread && currentThread.messages.length === 0) {
      const title = userMessage.slice(0, 50);
      setThreads(prev => prev.map(t => 
        t.id === activeChat ? { ...t, title } : t
      ));
    }

    setIsTyping(true);

    try {
      console.log('🌐 Calling API...');
      const response = await queryAPI(userMessage);

      console.log('✅ API Response:', {
        hasMessage: !!response.message,
        hasTableData: !!response.tableData,
        hasChart: !!response.chart,
        tableDataLength: response.tableData?.length
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.message || 'No response',
        role: 'assistant',
        timestamp: new Date().toString(),
        chart: response.chart,
        data: response.tableData, // ✅ FIXED - using tableData
        metadata: {
          outputFormat: response.output_format,
          outputConfidence: response.output_confidence,
          queryType: response.use_case,
          rowCount: response.row_count || 0,
          sqlSource: response.sql_source
        }
      };

      console.log('🤖 Assistant message created:', {
        hasChart: !!assistantMessage.chart,
        hasData: !!assistantMessage.data,
        dataLength: assistantMessage.data?.length
      });

      setThreads(prevThreads => {
        const updated = prevThreads.map(t => {
          if (t.id === activeChat) {
            const newMessages = [...t.messages, assistantMessage];
            console.log('✅ Assistant message added. Total messages:', newMessages.length);
            return {
              ...t,
              messages: newMessages,
              lastUpdated: new Date().toString()
            };
          }
          return t;
        });
        return updated;
      });

    } catch (error) {
      console.error('❌ Error:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        role: 'assistant',
        timestamp: new Date().toString()
      };

      setThreads(prev => prev.map(t => 
        t.id === activeChat 
          ? { ...t, messages: [...t.messages, errorMessage] }
          : t
      ));
    } finally {
      setIsTyping(false);
      console.log('=== SEND MESSAGE END ===\n');
    }
  };

  const handleNewChat = () => {
    const newThread: ChatThread = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      lastUpdated: new Date().toString()
    };
    
    setThreads(prev => [newThread, ...prev]);
    setActiveChat(newThread.id);
    console.log('🆕 New chat:', newThread.id);
  };

  const handleSelectChat = (chatId: string) => {
    setActiveChat(chatId);
    console.log('📌 Selected:', chatId);
  };

  const handleDeleteChat = (chatId: string) => {
    setThreads(prev => {
      const remaining = prev.filter(t => t.id !== chatId);
      if (activeChat === chatId) {
        setActiveChat(remaining.length > 0 ? remaining[0].id : null);
      }
      return remaining;
    });
  };

  const isWelcomeView = messages.length === 0;

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
        {/* Header with user info and logout */}
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

        <ChatInput
          onSendMessage={handleSendMessage}
          disabled={isTyping}
          isTyping={isTyping}
        />
      </div>
    </div>
  );
}
