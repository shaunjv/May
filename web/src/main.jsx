import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import Markdown from 'react-markdown';
import {ArrowUp, Check, ChevronDown, Copy, Folder, FolderOpen, MessageSquare, PanelLeft, Plus, ShieldCheck, Square, X, Sparkles, Code2, Files, Lightbulb, Moon, Sun} from 'lucide-react';
import './style.css';
import './reference-theme.css';

export function Welcome({disabled, onSend}) {
  const shortcuts = [
    {title: 'Ask a question', description: 'Explore an idea or get a clear explanation.', icon: Lightbulb, color: 'violet', prompt: 'What can you help me with?'},
    {title: 'Write some code', description: 'Turn a small idea into a useful example.', icon: Code2, color: 'rose', prompt: 'Write a Python example that counts odd numbers'},
    {title: 'Explore my files', description: 'Review a plan to list your workspace.', icon: Files, color: 'green', prompt: 'List the files in my workspace'},
  ];
  return <div className="welcome"><div className="welcome-mark"><Sparkles size={32}/></div><h1>Welcome to Personal Agent</h1><p>A space to think, create, and get things done.</p><div className="suggestions">{shortcuts.map(({title, description, icon: Icon, color, prompt}) => <button key={title} className={'shortcut ' + color} disabled={disabled} onClick={() => void onSend(prompt).catch(() => {})}><Icon size={22}/><strong>{title}</strong><span>{description}</span></button>)}</div></div>;
}

export async function api(path, body) {
  const response = await fetch('/api' + path, {method: body === undefined ? 'GET' : 'POST', headers: {'Content-Type': 'application/json', 'X-Agent-Client': 'local-web-v1'}, ...(body === undefined ? {} : {body: JSON.stringify(body)})});
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'The request failed. Please try again.');
  return data;
}

export function preferredWorkspace(conversation, defaultWorkspace) {
  return conversation?.workspace || defaultWorkspace || '';
}

export function Composer({onSend, disabled, onCancel}) {
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);
  const send = async () => {
    if (!text.trim() || disabled || sending) return;
    setSending(true);
    try {await onSend(text); setText('');} finally {setSending(false);}
  };
  return <div className="composer"><textarea aria-label="Message Personal Agent" placeholder="Ask anything, or tell me what to work on…" value={text} onChange={e => setText(e.target.value)} onKeyDown={e => {if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {e.preventDefault(); void send().catch(() => {});}}}/><div className="composer-bottom"><span><ShieldCheck size={15}/> You’re in control</span>{onCancel ? <button className="send" aria-label="Stop request" onClick={onCancel}><Square size={16}/></button> : <button className="send" aria-label="Send message" disabled={disabled || sending || !text.trim()} onClick={() => void send().catch(() => {})}><ArrowUp size={20}/></button>}</div></div>;
}

function CopyButton({text}) {
  const [copied, setCopied] = useState(false);
  return <button className="icon copy" aria-label="Copy response" title={copied ? 'Copied' : 'Copy'} onClick={async () => {try {await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1500);} catch {setCopied(false);}}}>{copied ? <Check size={15}/> : <Copy size={15}/>}</button>;
}

export function PlanCard({task, onAction, disabled}) {
  const waiting = task.state === 'AWAITING_APPROVAL';
  return <section className={'plan-card ' + task.state.toLowerCase()} aria-label="Task review"><header><ShieldCheck size={19}/><strong>{waiting ? 'Ready for your review' : task.state.replaceAll('_', ' ').toLowerCase()}</strong><span className="version">Plan v{task.package?.plan_version || 1}</span></header>
    {task.plan && <><h3>{task.plan.goal}</h3><p className="muted">Nothing runs until you approve these exact actions.</p>
    <ol className="plan-steps">{task.plan.phases.flatMap(p => p.steps).map(step => <li key={step.step_id}>{step.description}</li>)}</ol>
    <div className="action-preview">{task.package.actions.map((action, i) => <details key={action.action_id} open><summary><span className="action-number">{i + 1}</span><code>{action.tool_name}</code><ChevronDown size={14}/></summary><div className="action-content"><span className="path">{action.tool_input.path}</span>{action.tool_name === 'file_write' && <><p className="muted">{task.preconditions[action.tool_input.path] === null ? 'Create new file' : 'Replace existing file — review the complete replacement below'}</p><pre><code>{action.tool_input.content}</code></pre></>}</div></details>)}</div>
    <details className="identity"><summary>Approval identity</summary><code>{task.review_hash}</code></details></>}
    {task.error && <p role="alert" className="task-error">{task.error}</p>}
    {task.stage && task.state === 'EXECUTING' && <p role="status">{task.stage}</p>}
    {waiting && <footer><button className="primary" disabled={disabled} onClick={() => onAction(task, 'approve')}><Check size={16}/> Approve & run</button><button disabled={disabled} onClick={() => onAction(task, 'reject')}>Reject</button><button className="quiet" disabled={disabled} onClick={() => onAction(task, 'cancel')}>Cancel</button></footer>}
    {task.state === 'EXECUTING' && <button disabled={disabled || task.cancel_requested} onClick={() => onAction(task, 'cancel')}>{task.cancel_requested ? 'Stopping…' : 'Stop after current action'}</button>}
    {['FAILED', 'CANCELLED', 'REJECTED'].includes(task.state) && <button disabled={disabled} onClick={() => onAction(task, 'retry')}>Prepare a new request</button>}
  </section>;
}

export function App() {
  const [list, setList] = useState([]), [conversation, setConversation] = useState(null);
  const [status, setStatus] = useState(null), [error, setError] = useState('');
  const [modal, setModal] = useState(false), [workspace, setWorkspace] = useState('');
  const [choosingWorkspace, setChoosingWorkspace] = useState(false);
  const [busy, setBusy] = useState(false), [sidebar, setSidebar] = useState(true);
  const [dark, setDark] = useState(() => localStorage.getItem('personal-agent-theme') === 'dark');
  const selected = useRef(null), end = useRef(null), scroll = useRef(null), nearBottom = useRef(true);
  const hasActiveWork = conversation?.tasks.some(task => ['PLANNING', 'AWAITING_APPROVAL', 'EXECUTING'].includes(task.state));
  const refresh = async () => {
    const cid = selected.current;
    const [items, data] = await Promise.all([api('/conversations'), cid ? api('/conversations/' + cid) : Promise.resolve(null)]);
    setList(items);
    if (cid === selected.current) setConversation(data);
    return data;
  };
  useEffect(() => {
    let alive = true;
    const load = async () => {try {const [s, items] = await Promise.all([api('/status'), api('/conversations')]); if (!alive) return; setStatus(s); setWorkspace(s.default_workspace); setList(items); if (items.length) {selected.current = items[0].id; await refresh();}} catch(e) {if (alive) setError(e.message);}};
    void load();
    return () => {alive = false;};
  }, []);
  useEffect(() => {
    if (!hasActiveWork) return undefined;
    const timer = setInterval(() => {void refresh().catch(() => setError('Connection lost. Check that the local server is running.'));}, 1200);
    return () => clearInterval(timer);
  }, [hasActiveWork]);
  useEffect(() => {localStorage.setItem('personal-agent-theme', dark ? 'dark' : 'light');}, [dark]);
  const active = conversation?.tasks.find(t => ['PLANNING', 'AWAITING_APPROVAL', 'EXECUTING'].includes(t.state));
  useEffect(() => {if (nearBottom.current) end.current?.scrollIntoView?.({behavior: 'smooth'});}, [conversation?.messages.length, active?.state]);
  const perform = async fn => {setBusy(true); setError(''); try {await fn(); await refresh();} catch(e) {setError(e.message); throw e;} finally {setBusy(false);}};
  const openWorkspaceModal = () => {setWorkspace(preferredWorkspace(conversation, status?.default_workspace || workspace)); setModal(true);};
  const create = async e => {e?.preventDefault(); await perform(async () => {const c = await api('/conversations', {workspace}); selected.current = c.id; setConversation(c); setWorkspace(c.workspace); setModal(false);});};
  const chooseWorkspace = async () => {setChoosingWorkspace(true); setError(''); try {const result = await api('/workspace-picker', {initial_directory: workspace || null}); if (result.selected) setWorkspace(result.selected);} catch (e) {setError(e.message);} finally {setChoosingWorkspace(false);}};
  const send = async text => perform(async () => {let cid = selected.current; if (!cid) {const c = await api('/conversations', {workspace}); cid = c.id; selected.current = cid;} await api('/conversations/' + cid + '/messages', {content: text}); nearBottom.current = true;});
  const action = (task, verb) => {void perform(async () => {if (verb === 'retry') await api('/conversations/' + task.conversation_id + '/messages', {content: task.input}); else await api('/tasks/' + task.id + '/' + verb, verb === 'approve' ? {review_hash: task.review_hash} : {});}).catch(() => {});};
  const choose = async cid => {selected.current = cid; setConversation(null); nearBottom.current = true; const data = await refresh(); setWorkspace(preferredWorkspace(data, status?.default_workspace || workspace));};
  return <div className={'app ' + (!sidebar ? 'collapsed ' : '') + (dark ? 'theme-dark' : '')}>
    <aside className="sidebar"><div className="brand"><span className="brand-mark">✳</span> Personal Agent<button className="icon" aria-label="Close sidebar" onClick={() => setSidebar(false)}><PanelLeft size={18}/></button></div><button className="new-chat" onClick={openWorkspaceModal}><Plus size={17}/> New conversation</button><div className="section-label">YOUR CONVERSATIONS</div><nav>{list.length ? list.map(c => <button key={c.id} className={conversation?.id === c.id ? 'selected' : ''} onClick={() => void choose(c.id).catch(e => setError(e.message))}><MessageSquare size={16}/><span>{c.title}</span></button>) : <p className="sidebar-empty">A little space for your next idea.</p>}</nav><div className="sidebar-foot"><span className="local-dot"/> Runs on your computer<p>Your API key stays on the server.</p></div></aside>
    <main><div className="topbar"><div><button className="icon" aria-label="Toggle sidebar" onClick={() => setSidebar(!sidebar)}><PanelLeft size={19}/></button><span>Personal Agent <span className="beta">LOCAL</span></span></div><div className="top-actions"><button className="icon theme-toggle" aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'} title={dark ? 'Light mode' : 'Dark mode'} onClick={() => setDark(!dark)}>{dark ? <Sun size={18}/> : <Moon size={18}/>}</button><button className="workspace" title={conversation?.workspace || workspace} onClick={openWorkspaceModal}><Folder size={15}/><span>{(conversation?.workspace || workspace).split(/[\\/]/).pop() || 'Choose workspace'}</span><ChevronDown size={14}/></button></div></div>
    {error && <div className="banner" role="alert">{error}<button className="icon" aria-label="Dismiss error" onClick={() => setError('')}><X size={16}/></button></div>}
    {status && !status.provider_ready && <div className="banner">Add NVIDIA_API_KEY and NVIDIA_MODEL to your local .env file, then restart the server. Never enter the key in chat.</div>}
    <div className="thread" ref={scroll} onScroll={() => {const el = scroll.current; nearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 100;}}><div className="thread-inner">
      {!conversation?.messages.length ? <Welcome disabled={busy || !!active || !status?.provider_ready} onSend={send}/> : conversation.messages.map(message => <article key={message.id} className={'message ' + message.role}><div className="message-label">{message.role === 'user' ? 'You' : <><span>✳</span> Personal Agent</>}</div><div className="message-body"><Markdown components={{a: ({children, ...props}) => <a {...props} target="_blank" rel="noopener noreferrer">{children}</a>, img: () => <span>[Image omitted]</span>}}>{message.content}</Markdown></div>{message.role === 'assistant' && <CopyButton text={message.content}/>}</article>)}
      {conversation?.tasks.filter(t => t.kind === 'task' || t.state === 'FAILED').map(task => <PlanCard key={task.id} task={task} onAction={action} disabled={busy || (!!active && active.id !== task.id)}/>)}
      {active?.state === 'PLANNING' && <div className="thinking" role="status"><span className="pulse"/> {active.stage}</div>}<div ref={end}/>
    </div></div>
    <div className="composer-wrap"><Composer onSend={send} disabled={busy || !!active || !status?.provider_ready} onCancel={active?.state === 'PLANNING' ? () => action(active, 'cancel') : null}/><p className="composer-note">{active?.state === 'AWAITING_APPROVAL' ? 'Review the plan above to continue.' : 'Enter to send · Shift + Enter for a new line · File actions need your approval'}</p></div></main>
    {modal && <div className="modal-backdrop"><form className="modal" onSubmit={e => void create(e).catch(() => {})}><header><h2>Start a conversation</h2><button type="button" className="icon" aria-label="Close dialog" onClick={() => setModal(false)}><X size={20}/></button></header><p>Choose the local folder this conversation can work in. Each conversation keeps its own workspace.</p><label htmlFor="workspace">Workspace folder</label><div className="workspace-input-row"><input autoFocus id="workspace" value={workspace} onChange={e => setWorkspace(e.target.value)} placeholder="E:\projects\my-project"/><button type="button" className="choose-folder" disabled={busy || choosingWorkspace} onClick={() => void chooseWorkspace()}>{choosingWorkspace ? 'Opening…' : <><FolderOpen size={16}/> Choose folder</>}</button></div><small>Use the folder picker or paste an existing folder path. Changing workspace starts a new conversation; it never changes an approved task.</small><button className="primary" disabled={busy || choosingWorkspace || !workspace.trim()} type="submit">Create conversation <Plus size={16}/></button></form></div>}
  </div>;
}

if (document.getElementById('root')) createRoot(document.getElementById('root')).render(<App/>);
