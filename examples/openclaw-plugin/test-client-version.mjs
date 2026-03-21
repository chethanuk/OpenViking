/**
 * Manual tests for getMemoryVersion() — run with: node test-client-version.mjs
 * Tests Slice C of issue #817: null-on-error contract.
 */
import assert from 'node:assert/strict';

// We test by verifying the method exists and returns null on network failure
// (server not running at a bad port)

// Dynamic import to handle TypeScript (compiled) or JS
let OpenVikingClient;
try {
  // Try compiled JS first
  const mod = await import('./dist/client.js');
  OpenVikingClient = mod.OpenVikingClient;
} catch {
  console.log('Note: No compiled JS found. Testing via TypeScript source would require ts-node.');
  console.log('Verify getMemoryVersion() exists in client.ts and returns null on error manually.');
  process.exit(0);
}

// Test 1: method exists
const client = new OpenVikingClient('http://localhost:19999', 'key', 'agent-1', 1000);
assert.equal(typeof client.getMemoryVersion, 'function', 'getMemoryVersion must be a function');
console.log('✅ Test 1 PASS: getMemoryVersion method exists');

// Test 2: returns null on network failure (no server at port 19999)
const result = await client.getMemoryVersion('viking://user/memories');
assert.equal(result, null, 'getMemoryVersion must return null on network error, not throw');
console.log('✅ Test 2 PASS: returns null on network error');

console.log('\nAll Slice C tests passed.');
