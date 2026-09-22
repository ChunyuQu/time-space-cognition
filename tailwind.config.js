/** @type {import('tailwindcss').Config} */
module.exports = {
  // Templates and content both carry class names, so both must be scanned.
  content: [
    './layouts/**/*.html',
    './content/**/*.md',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#0f172a',
        secondary: '#1e293b',
        accent: '#3b82f6',
      },
    },
  },
  plugins: [],
}
