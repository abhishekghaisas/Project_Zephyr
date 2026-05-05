import { queryAPI } from "./api/query";
import { useState, useEffect, useRef } from 'react';
import { ChatSidebar } from './components/ChatSidebar';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { ChatWelcome } from './components/ChatWelcome';
import { Login } from './components/Login';
import { AuthProvider, useAuth } from '.useAuth';
import { ScrollArea } from './components/ui/scroll-area';
import { Button } from './components/ui/button';
import { LogOut } from 'lucide-react';

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

function AppContent() {
  const { isAuthenticated, user, login, logout } = useAuth();
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [pendingThread, setPendingThread] = useState<ChatThread | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [activeChat, setActiveChat] = useState<string | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Load threads from localStorage
  useEffect(() => {
    if (!isAuthenticated) return;
    
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const parsedThreads = JSON.parse(stored);
        const threadsWithDates = parsedThreads.map((thread: ChatThread) => ({
          ...thread,
          lastUpdated: new Date(thread.lastUpdated),
          messages: thread.messages.map((msg: Message) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }))
        }));
        setThreads(threadsWithDates);
        if (threadsWithDates.length > 0 && !activeChat) {
          setActiveChat(threadsWithDates[0].id);
        }
      } catch (error) {
        console.error('Failed to load chat threads:', error);
      }
    }
  }, [isAuthenticated]);

  // Save threads to localStorage
  useEffect(() => {
    if (!isAuthenticated || threads.length === 0) return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
  }, [threads, isAuthenticated]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [activeThread?.messages.length, isTyping]);

  const activeThread = activeChat 
    ? threads.find(t => t.id === activeChat) 
    : pendingThread;

  const handleNewChat = () => {
    const newThread: ChatThread = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      lastUpdated: new Date()
    };
    setPendingThread(newThread);
    setActiveChat(null);
  };

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      role: 'user',
      timestamp: new Date()
    };

    let currentThread = activeThread;
    
    if (!currentThread) {
      const newThread: ChatThread = {
        id: Date.now().toString(),
        title: content.slice(0, 50),
        messages: [userMessage],
        lastUpdated: new Date()
      };
      currentThread = newThread;
      setThreads(prev => [newThread, ...prev]);
      setActiveChat(newThread.id);
      setPendingThread(null);
    } else {
      if (pendingThread) {
        const updatedPending = {
          ...pendingThread,
          title: content.slice(0, 50),
          messages: [userMessage],
          lastUpdated: new Date()
        };
        setThreads(prev => [updatedPending, ...prev]);
        setActiveChat(updatedPending.id);
        setPendingThread(null);
        currentThread = updatedPending;
      } else {
        setThreads(prev => prev.map(t => 
          t.id === currentThread!.id 
            ? { ...t, messages: [...t.messages, userMessage], lastUpdated: new Date() }
            : t
        ));
      }
    }

    setIsTyping(true);

    try {
      const response = await queryAPI(content);
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.message || response.insights?.join('\n\n') || 'Query executed successfully.',
        role: 'assistant',
        timestamp: new Date(),
        chart: response.chart || null,
        data: response.data || null
      };

      setThreads(prev => prev.map(t => 
        t.id === currentThread!.id
          ? { ...t, messages: [...t.messages, assistantMessage], lastUpdated: new Date() }
          : t
      ));
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        role: 'assistant',
        timestamp: new Date()
      };

      setThreads(prev => prev.map(t => 
        t.id === currentThread!.id
          ? { ...t, messages: [...t.messages, errorMessage], lastUpdated: new Date() }
          : t
      ));
    } finally {
      setIsTyping(false);
    }
  };

  const handleDeleteThread = (threadId: string) => {
    setThreads(prev => prev.filter(t => t.id !== threadId));
    if (activeChat === threadId) {
      setActiveChat(threads[0]?.id || null);
      setPendingThread(null);
    }
  };

  const handleSelectThread = (threadId: string) => {
    setActiveChat(threadId);
    setPendingThread(null);
  };

  // Show login page if not authenticated
  if (!isAuthenticated) {
    return <Login onLogin={login} />;
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <ChatSidebar
        threads={threads}
        activeThreadId={activeChat}
        onSelectThread={handleSelectThread}
        onNewChat={handleNewChat}
        onDeleteThread={handleDeleteThread}
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
      />

      <div className="flex-1 flex flex-col min-w-0">
        {/* Header with Logout */}
        <div className="border-b border-gray-200 bg-white px-6 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-gray-900">AIport Assistant</h1>
            <p className="text-xs text-gray-500">Logged in as {user}</p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={logout}
            className="h-8 px-3"
          >
            <LogOut className="h-4 w-4 mr-2" />
            Logout
          </Button>
        </div>

        {/* Chat Area */}
        <ScrollArea ref={scrollAreaRef} className="flex-1">
          <div className="max-w-4xl mx-auto">
            {activeThread?.messages.length === 0 ? (
              <ChatWelcome />
            ) : (
              <div className="py-4">
                {activeThread?.messages.map((message) => (
                  <ChatMessage key={message.id} message={message} />
                ))}
                {isTyping && (
                  <div className="flex gap-4 p-6 bg-gray-50/30">
                    <div className="flex-shrink-0">
                      <div className="h-8 w-8 bg-green-100 rounded-full flex items-center justify-center">
                        <span className="text-sm font-medium text-green-700">AI</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 pt-1">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </ScrollArea>

        <ChatInput onSendMessage={handleSendMessage} disabled={isTyping} />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
