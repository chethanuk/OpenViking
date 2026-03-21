/**
 * Tests for Slice D: version cache hit/miss/fallback logic
 * Run: node test-plugin-version-cache.mjs
 */
import assert from 'node:assert/strict';

// These tests validate the LOGIC of the version cache decision,
// not the hook itself (which requires a running server).

// Test 1: Cache HIT — find() is NOT called when version matches
{
  let findCallCount = 0;
  const mockClient = {
    async getMemoryVersion(_ns) { return 7; },
    async find(_q, _opts) { findCallCount++; return { memories: [], total: 0 }; },
  };
  const versionCache = new Map();
  const sessionId = 'session-abc';
  const targetUri = 'viking://user/memories';
  const cacheKey = `${sessionId}:${targetUri}`;
  versionCache.set(cacheKey, 7);

  const serverVersion = await mockClient.getMemoryVersion(targetUri);
  const cached = versionCache.get(cacheKey);
  const shouldSkip = serverVersion !== null && serverVersion === cached;
  if (!shouldSkip) {
    await mockClient.find('query', { targetUri });
    versionCache.set(cacheKey, serverVersion);
  }

  assert.equal(findCallCount, 0, 'find() must NOT be called on cache hit');
  console.log('✅ Test 1 PASS: find() skipped on version cache hit');
}

// Test 2: Cache MISS — find() IS called when version changes
{
  let findCallCount = 0;
  const mockClient = {
    async getMemoryVersion(_ns) { return 8; },
    async find(_q, _opts) { findCallCount++; return { memories: [], total: 0 }; },
  };
  const versionCache = new Map();
  versionCache.set('session-abc:viking://user/memories', 7);

  const serverVersion = await mockClient.getMemoryVersion('viking://user/memories');
  const cached = versionCache.get('session-abc:viking://user/memories');
  const shouldSkip = serverVersion !== null && serverVersion === cached;
  if (!shouldSkip) {
    await mockClient.find('query', {});
    versionCache.set('session-abc:viking://user/memories', serverVersion);
  }

  assert.equal(findCallCount, 1, 'find() must be called on version change');
  assert.equal(versionCache.get('session-abc:viking://user/memories'), 8, 'cache updated');
  console.log('✅ Test 2 PASS: find() called on version mismatch');
}

// Test 3: FALLBACK — find() called when getMemoryVersion() returns null
{
  let findCallCount = 0;
  const mockClient = {
    async getMemoryVersion(_ns) { return null; },
    async find(_q, _opts) { findCallCount++; return { memories: [], total: 0 }; },
  };
  const versionCache = new Map();
  versionCache.set('session-abc:viking://user/memories', 7);

  const serverVersion = await mockClient.getMemoryVersion('viking://user/memories');
  const cached = versionCache.get('session-abc:viking://user/memories');
  const shouldSkip = serverVersion !== null && serverVersion === cached;
  if (!shouldSkip) {
    await mockClient.find('query', {});
    // Do NOT update cache when serverVersion is null
  }

  assert.equal(findCallCount, 1, 'find() must be called when getMemoryVersion() returns null');
  assert.equal(versionCache.get('session-abc:viking://user/memories'), 7, 'cache NOT updated on null');
  console.log('✅ Test 3 PASS: fallback to full recall when version endpoint fails');
}

// Test 4: First call (no cache entry) — find() IS called
{
  let findCallCount = 0;
  const mockClient = {
    async getMemoryVersion(_ns) { return 3; },
    async find(_q, _opts) { findCallCount++; return { memories: [], total: 0 }; },
  };
  const versionCache = new Map();
  const serverVersion = await mockClient.getMemoryVersion('viking://user/memories');
  const cached = versionCache.get('new-session:viking://user/memories');
  const shouldSkip = serverVersion !== null && serverVersion === cached;
  if (!shouldSkip) {
    await mockClient.find('query', {});
    versionCache.set('new-session:viking://user/memories', serverVersion);
  }

  assert.equal(findCallCount, 1, 'find() must be called on first request');
  assert.equal(versionCache.get('new-session:viking://user/memories'), 3, 'cache seeded');
  console.log('✅ Test 4 PASS: find() called on first request, cache seeded');
}

// Test 5: No session ID — find() is ALWAYS called
{
  let findCallCount = 0;
  const mockClient = {
    async getMemoryVersion(_ns) { return 7; },
    async find(_q, _opts) { findCallCount++; return { memories: [], total: 0 }; },
  };
  const versionCache = new Map();
  const hookSessionId = '';  // empty string

  if (hookSessionId) {
    const serverVersion = await mockClient.getMemoryVersion('viking://user/memories');
    const cached = versionCache.get(`${hookSessionId}:viking://user/memories`);
    if (serverVersion !== null && serverVersion === cached) {
      // skip
    } else {
      await mockClient.find('query', {});
    }
  } else {
    await mockClient.find('query', {});
  }

  assert.equal(findCallCount, 1, 'find() must be called when sessionId is empty');
  console.log('✅ Test 5 PASS: find() always called when sessionId is absent');
}

console.log('\nAll Slice D logic tests passed.');
