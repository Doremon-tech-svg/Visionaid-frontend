import { useState } from 'react';
import { X, LogIn, UserPlus } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000';

export default function AuthModal({ onClose, onAuthed, embedded = false }) {
  const [mode, setMode] = useState('login'); // 'login' | 'signup'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    setBusy(true);

    try {
      const path =
        mode === 'login'
          ? '/api/auth/login'
          : '/api/auth/signup';

      const body =
        mode === 'login'
          ? { email, password }
          : { email, password, name };

      const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      const data = await res.json();

      if (!data.success) {
        setError(data.error || 'Something went wrong');
        return;
      }

      localStorage.setItem('visionaid_token', data.token);
      localStorage.setItem('visionaid_user', JSON.stringify(data.user));

      onAuthed(data.user);
    } catch {
      setError('Could not reach the server');
    } finally {
      setBusy(false);
    }
  };

  const card = (
    <div
      className="bg-stone-950 border border-amber-400/40 rounded-2xl w-full max-w-sm shadow-2xl"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between px-6 py-4 border-b border-stone-800">
        <h3 className="text-amber-400 text-sm font-bold">
          {mode === 'login' ? 'LOG IN' : 'CREATE ACCOUNT'}
        </h3>

        {!embedded && (
          <button
            onClick={onClose}
            aria-label="Close"
            className="text-stone-500 hover:text-amber-300"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      <form onSubmit={submit} className="p-6 space-y-4">
        {mode === 'signup' && (
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Name (optional)"
            className="w-full bg-black border border-stone-700 text-stone-200 px-3 py-3 rounded-lg text-sm focus:border-amber-400 outline-none"
          />
        )}

        <input
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          autoComplete="email"
          className="w-full bg-black border border-stone-700 text-stone-200 px-3 py-3 rounded-lg text-sm focus:border-amber-400 outline-none"
        />

        <input
          type="password"
          required
          minLength={6}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password (6+ characters)"
          autoComplete={
            mode === 'login'
              ? 'current-password'
              : 'new-password'
          }
          className="w-full bg-black border border-stone-700 text-stone-200 px-3 py-3 rounded-lg text-sm focus:border-amber-400 outline-none"
        />

        {error && (
          <p className="text-red-400 text-xs">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy}
          className="w-full bg-amber-400 hover:bg-amber-300 disabled:opacity-50 text-black py-3 rounded-lg font-bold text-sm flex items-center justify-center gap-2"
        >
          {mode === 'login' ? (
            <LogIn className="w-4 h-4" />
          ) : (
            <UserPlus className="w-4 h-4" />
          )}

          {busy
            ? 'PLEASE WAIT...'
            : mode === 'login'
              ? 'LOG IN'
              : 'SIGN UP'}
        </button>

        <button
          type="button"
          onClick={() => {
            setMode((m) =>
              m === 'login' ? 'signup' : 'login'
            );
            setError(null);
          }}
          className="w-full text-center text-stone-500 hover:text-amber-300 text-xs underline transition-colors"
        >
          {mode === 'login'
            ? "Don't have an account? Sign up"
            : 'Already have an account? Log in'}
        </button>
      </form>
    </div>
  );

  // Used by the mandatory login screen in App.jsx
  if (embedded) {
    return card;
  }

  // Used when AuthModal is opened as a normal modal
  return (
    <div
      className="fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-[70]"
      onClick={onClose}
    >
      {card}
    </div>
  );
}