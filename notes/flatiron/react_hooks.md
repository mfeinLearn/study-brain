# React hooks: useEffect vs useMemo — SAMPLE NOTES

- `useEffect` runs side effects after render: fetching data, subscriptions,
  touching the DOM. It does not return a value to the render.
- `useMemo` caches a computed value between renders and recomputes it only
  when its dependencies change. Use it for expensive calculations, not for
  side effects.
