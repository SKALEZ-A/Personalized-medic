/**
 * Accessibility Manager Component for AI Personalized Medicine Platform
 * Comprehensive accessibility features and screen reader support
 */

import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import './AccessibilityManager.css';

// Accessibility Context
const AccessibilityContext = createContext();

export const useAccessibility = () => {
  const context = useContext(AccessibilityContext);
  if (!context) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider');
  }
  return context;
};

// Accessibility Provider Component
export const AccessibilityProvider = ({ children }) => {
  const [preferences, setPreferences] = useState({
    highContrast: false,
    largeText: false,
    reducedMotion: false,
    screenReader: false,
    keyboardNavigation: true,
    focusVisible: true,
    colorBlindMode: 'none', // 'none', 'protanopia', 'deuteranopia', 'tritanopia'
    speechEnabled: false,
    language: 'en'
  });

  const [announcements, setAnnouncements] = useState([]);
  const [focusHistory, setFocusHistory] = useState([]);
  const speechSynthesisRef = useRef(null);

  // Load preferences from localStorage
  useEffect(() => {
    const savedPreferences = localStorage.getItem('accessibility-preferences');
    if (savedPreferences) {
      try {
        const parsed = JSON.parse(savedPreferences);
        setPreferences(prev => ({ ...prev, ...parsed }));
      } catch (error) {
        console.warn('Failed to parse accessibility preferences:', error);
      }
    }

    // Detect system preferences
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPreferences(prev => ({
      ...prev,
      reducedMotion: mediaQuery.matches
    }));

    const contrastQuery = window.matchMedia('(prefers-contrast: high)');
    setPreferences(prev => ({
      ...prev,
      highContrast: contrastQuery.matches
    }));

    // Listen for preference changes
    const handleMotionChange = (e) => {
      setPreferences(prev => ({ ...prev, reducedMotion: e.matches }));
    };

    const handleContrastChange = (e) => {
      setPreferences(prev => ({ ...prev, highContrast: e.matches }));
    };

    mediaQuery.addEventListener('change', handleMotionChange);
    contrastQuery.addEventListener('change', handleContrastChange);

    return () => {
      mediaQuery.removeEventListener('change', handleMotionChange);
      contrastQuery.removeEventListener('change', handleContrastChange);
    };
  }, []);

  // Save preferences to localStorage
  useEffect(() => {
    localStorage.setItem('accessibility-preferences', JSON.stringify(preferences));

    // Apply preferences to document
    document.documentElement.setAttribute('data-high-contrast', preferences.highContrast);
    document.documentElement.setAttribute('data-large-text', preferences.largeText);
    document.documentElement.setAttribute('data-reduced-motion', preferences.reducedMotion);
    document.documentElement.setAttribute('data-color-blind', preferences.colorBlindMode);
    document.documentElement.setAttribute('lang', preferences.language);

    if (preferences.focusVisible) {
      document.documentElement.classList.add('focus-visible');
    } else {
      document.documentElement.classList.remove('focus-visible');
    }
  }, [preferences]);

  // Initialize speech synthesis
  useEffect(() => {
    if ('speechSynthesis' in window) {
      speechSynthesisRef.current = window.speechSynthesis;
    }
  }, []);

  // Announce changes to screen readers
  const announce = (message, priority = 'polite') => {
    const announcement = {
      id: Date.now(),
      message,
      priority,
      timestamp: new Date()
    };

    setAnnouncements(prev => [...prev, announcement]);

    // Remove announcement after it's been processed
    setTimeout(() => {
      setAnnouncements(prev => prev.filter(a => a.id !== announcement.id));
    }, 1000);

    // Speak if speech is enabled
    if (preferences.speechEnabled && speechSynthesisRef.current) {
      const utterance = new SpeechSynthesisUtterance(message);
      speechSynthesisRef.current.speak(utterance);
    }
  };

  // Update preferences
  const updatePreference = (key, value) => {
    setPreferences(prev => ({ ...prev, [key]: value }));
    announce(`${key.replace(/([A-Z])/g, ' $1').toLowerCase()} ${value ? 'enabled' : 'disabled'}`);
  };

  // Focus management
  const setFocus = (elementId, announceFocus = true) => {
    const element = document.getElementById(elementId);
    if (element) {
      element.focus();
      setFocusHistory(prev => [...prev.slice(-9), elementId]); // Keep last 10

      if (announceFocus) {
        const label = element.getAttribute('aria-label') ||
                     element.getAttribute('aria-labelledby') ||
                     element.textContent?.slice(0, 50) ||
                     'Element';
        announce(`Focused on ${label}`);
      }
    }
  };

  // Trap focus within a container
  const trapFocus = (containerRef, initialFocusId = null) => {
    const container = containerRef.current;
    if (!container) return () => {};

    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    // Focus initial element
    if (initialFocusId) {
      setFocus(initialFocusId, false);
    } else if (firstElement) {
      firstElement.focus();
    }

    const handleKeyDown = (e) => {
      if (e.key === 'Tab') {
        if (e.shiftKey) {
          // Shift + Tab
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement?.focus();
          }
        } else {
          // Tab
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement?.focus();
          }
        }
      }

      if (e.key === 'Escape') {
        // Allow escape to close modal/container
        const closeButton = container.querySelector('[data-close], .close-btn, [aria-label*="close"]');
        if (closeButton) {
          closeButton.click();
        }
      }
    };

    container.addEventListener('keydown', handleKeyDown);

    return () => {
      container.removeEventListener('keydown', handleKeyDown);
    };
  };

  // Skip links management
  const createSkipLink = (targetId, label) => {
    return {
      href: `#${targetId}`,
      onClick: (e) => {
        e.preventDefault();
        setFocus(targetId);
      },
      children: label,
      className: 'skip-link'
    };
  };

  // ARIA live region for announcements
  const LiveAnnouncer = () => (
    <div aria-live="polite" aria-atomic="true" className="sr-only">
      {announcements.map(announcement => (
        <div key={announcement.id}>{announcement.message}</div>
      ))}
    </div>
  );

  // Keyboard shortcut handler
  const handleKeyboardShortcut = (key, callback, ctrlKey = false) => {
    useEffect(() => {
      const handleKeyDown = (e) => {
        if (e.key === key && (!ctrlKey || e.ctrlKey)) {
          e.preventDefault();
          callback();
        }
      };

      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }, [key, ctrlKey, callback]);
  };

  const contextValue = {
    preferences,
    updatePreference,
    announce,
    setFocus,
    trapFocus,
    createSkipLink,
    focusHistory,
    handleKeyboardShortcut,
    LiveAnnouncer
  };

  return (
    <AccessibilityContext.Provider value={contextValue}>
      <LiveAnnouncer />
      {children}
    </AccessibilityContext.Provider>
  );
};

// Accessibility Toolbar Component
export const AccessibilityToolbar = () => {
  const { preferences, updatePreference } = useAccessibility();
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className={`accessibility-toolbar ${isExpanded ? 'expanded' : ''}`}>
      <button
        className="accessibility-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-label="Accessibility options"
        aria-expanded={isExpanded}
      >
        ♿
      </button>

      {isExpanded && (
        <div className="accessibility-panel" role="dialog" aria-label="Accessibility settings">
          <h3>Accessibility Settings</h3>

          <div className="accessibility-options">
            <label className="option-item">
              <input
                type="checkbox"
                checked={preferences.highContrast}
                onChange={(e) => updatePreference('highContrast', e.target.checked)}
              />
              <span>High Contrast</span>
            </label>

            <label className="option-item">
              <input
                type="checkbox"
                checked={preferences.largeText}
                onChange={(e) => updatePreference('largeText', e.target.checked)}
              />
              <span>Large Text</span>
            </label>

            <label className="option-item">
              <input
                type="checkbox"
                checked={preferences.reducedMotion}
                onChange={(e) => updatePreference('reducedMotion', e.target.checked)}
              />
              <span>Reduced Motion</span>
            </label>

            <label className="option-item">
              <input
                type="checkbox"
                checked={preferences.speechEnabled}
                onChange={(e) => updatePreference('speechEnabled', e.target.checked)}
              />
              <span>Text-to-Speech</span>
            </label>

            <label className="option-item">
              <input
                type="checkbox"
                checked={preferences.focusVisible}
                onChange={(e) => updatePreference('focusVisible', e.target.checked)}
              />
              <span>Visible Focus Indicators</span>
            </label>

            <div className="option-item">
              <label htmlFor="color-blind-mode">Color Blind Mode:</label>
              <select
                id="color-blind-mode"
                value={preferences.colorBlindMode}
                onChange={(e) => updatePreference('colorBlindMode', e.target.value)}
              >
                <option value="none">None</option>
                <option value="protanopia">Protanopia</option>
                <option value="deuteranopia">Deuteranopia</option>
                <option value="tritanopia">Tritanopia</option>
              </select>
            </div>

            <div className="option-item">
              <label htmlFor="language-select">Language:</label>
              <select
                id="language-select"
                value={preferences.language}
                onChange={(e) => updatePreference('language', e.target.value)}
              >
                <option value="en">English</option>
                <option value="es">Español</option>
                <option value="fr">Français</option>
                <option value="de">Deutsch</option>
              </select>
            </div>
          </div>

          <div className="accessibility-actions">
            <button
              className="reset-btn"
              onClick={() => {
                const defaults = {
                  highContrast: false,
                  largeText: false,
                  reducedMotion: false,
                  screenReader: false,
                  keyboardNavigation: true,
                  focusVisible: true,
                  colorBlindMode: 'none',
                  speechEnabled: false,
                  language: 'en'
                };
                Object.entries(defaults).forEach(([key, value]) => {
                  updatePreference(key, value);
                });
              }}
            >
              Reset to Defaults
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// Accessible Form Field Component
export const AccessibleField = ({
  label,
  id,
  type = 'text',
  value,
  onChange,
  error,
  hint,
  required = false,
  children,
  ...props
}) => {
  const fieldId = id || `field-${Math.random().toString(36).substr(2, 9)}`;
  const errorId = `${fieldId}-error`;
  const hintId = `${fieldId}-hint`;

  const hasError = Boolean(error);
  const hasHint = Boolean(hint);

  return (
    <div className={`accessible-field ${hasError ? 'has-error' : ''}`}>
      <label htmlFor={fieldId} className="field-label">
        {label}
        {required && <span className="required-indicator" aria-label="required">*</span>}
      </label>

      {hasHint && (
        <div id={hintId} className="field-hint">
          {hint}
        </div>
      )}

      {children ? (
        React.cloneElement(children, {
          id: fieldId,
          'aria-describedby': [hasError && errorId, hasHint && hintId].filter(Boolean).join(' ') || undefined,
          'aria-invalid': hasError,
          'aria-required': required,
          ...props
        })
      ) : (
        <input
          id={fieldId}
          type={type}
          value={value}
          onChange={onChange}
          aria-describedby={[hasError && errorId, hasHint && hintId].filter(Boolean).join(' ') || undefined}
          aria-invalid={hasError}
          aria-required={required}
          {...props}
        />
      )}

      {hasError && (
        <div id={errorId} className="field-error" role="alert">
          {error}
        </div>
      )}
    </div>
  );
};

// Accessible Modal Component
export const AccessibleModal = ({
  isOpen,
  onClose,
  title,
  children,
  size = 'medium',
  closeOnEscape = true,
  closeOnOverlayClick = true
}) => {
  const { trapFocus, announce } = useAccessibility();
  const modalRef = useRef(null);
  const previousFocusRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      previousFocusRef.current = document.activeElement;
      announce(`Opened ${title} dialog`);
      const cleanup = trapFocus(modalRef, `${title.toLowerCase().replace(/\s+/g, '-')}-content`);

      const handleEscape = (e) => {
        if (e.key === 'Escape' && closeOnEscape) {
          onClose();
        }
      };

      document.addEventListener('keydown', handleEscape);

      // Prevent body scroll
      document.body.style.overflow = 'hidden';

      return () => {
        cleanup();
        document.removeEventListener('keydown', handleEscape);
        document.body.style.overflow = 'unset';

        // Restore focus
        if (previousFocusRef.current && typeof previousFocusRef.current.focus === 'function') {
          previousFocusRef.current.focus();
        }
      };
    }
  }, [isOpen, onClose, title, trapFocus, announce, closeOnEscape]);

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby={`${title.toLowerCase().replace(/\s+/g, '-')}-title`}>
      <div className="modal-backdrop" onClick={closeOnOverlayClick ? onClose : undefined} />

      <div
        ref={modalRef}
        className={`accessible-modal ${size}`}
        role="document"
      >
        <header className="modal-header">
          <h2 id={`${title.toLowerCase().replace(/\s+/g, '-')}-title`} className="modal-title">
            {title}
          </h2>
          <button
            className="modal-close"
            onClick={onClose}
            aria-label="Close dialog"
            data-close
          >
            ✕
          </button>
        </header>

        <div id={`${title.toLowerCase().replace(/\s+/g, '-')}-content`} className="modal-content">
          {children}
        </div>
      </div>
    </div>
  );
};

// Accessible Table Component
export const AccessibleTable = ({
  data,
  columns,
  caption,
  sortable = false,
  onSort = null
}) => {
  const [sortColumn, setSortColumn] = useState(null);
  const [sortDirection, setSortDirection] = useState('asc');

  const handleSort = (column) => {
    if (!sortable || !onSort) return;

    const newDirection = sortColumn === column && sortDirection === 'asc' ? 'desc' : 'asc';
    setSortColumn(column);
    setSortDirection(newDirection);
    onSort(column, newDirection);
  };

  return (
    <table className="accessible-table" role="table">
      {caption && (
        <caption className="sr-only">{caption}</caption>
      )}

      <thead>
        <tr role="row">
          {columns.map((column, index) => (
            <th
              key={column.key}
              scope="col"
              role="columnheader"
              aria-sort={
                sortable && sortColumn === column.key
                  ? (sortDirection === 'asc' ? 'ascending' : 'descending')
                  : 'none'
              }
              className={sortable && onSort ? 'sortable' : ''}
              onClick={() => handleSort(column.key)}
              tabIndex={sortable && onSort ? 0 : -1}
            >
              {column.label}
              {sortable && onSort && sortColumn === column.key && (
                <span className="sort-indicator" aria-hidden="true">
                  {sortDirection === 'asc' ? ' ↑' : ' ↓'}
                </span>
              )}
            </th>
          ))}
        </tr>
      </thead>

      <tbody>
        {data.map((row, rowIndex) => (
          <tr key={row.id || rowIndex} role="row">
            {columns.map((column) => (
              <td key={column.key} role="gridcell">
                {column.render ? column.render(row[column.key], row) : row[column.key]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
};

// Screen Reader Instructions Component
export const ScreenReaderInstructions = () => (
  <div className="sr-only">
    <h2>Keyboard Navigation Instructions</h2>
    <ul>
      <li>Use Tab to navigate through interactive elements</li>
      <li>Use Shift+Tab to navigate backwards</li>
      <li>Use Enter or Space to activate buttons and links</li>
      <li>Use Escape to close modals and menus</li>
      <li>Use arrow keys to navigate within menus and lists</li>
      <li>Use Ctrl+/ to focus the search input</li>
    </ul>

    <h2>Screen Reader Features</h2>
    <ul>
      <li>All images have alternative text descriptions</li>
      <li>Form fields have associated labels and instructions</li>
      <li>Status changes are announced automatically</li>
      <li>Color is never the only way information is conveyed</li>
      <li>Focus indicators are always visible</li>
    </ul>
  </div>
);

export default AccessibilityProvider;
