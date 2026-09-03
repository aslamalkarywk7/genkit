/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  GenkitError,
  defineDynamicActionProvider,
  z,
  type PluginProvider,
} from '@genkit-ai/core';
import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import {
  GenerateResponse,
  GenerationAbortedError,
  GenerationResponseError,
  generate,
  generateStream,
  normalizeMiddleware,
  toGenerateActionOptions,
  toGenerateRequest,
  type GenerateOptions,
} from '../../src/generate.js';
import { generateMiddleware } from '../../src/generate/middleware.js';
import {
  defineModel,
  type GenerateResponseData,
  type ModelAction,
  type ModelMiddleware,
  type ModelMiddlewareWithOptions,
} from '../../src/model.js';
import { defineResource } from '../../src/resource.js';
import { ToolInterruptError, defineTool, tool } from '../../src/tool.js';

initNodeFeatures();

describe('toGenerateRequest', () => {
  const registry = new Registry();
  // register tools
  const tellAFunnyJoke = defineTool(
    registry,
    {
      name: 'tellAFunnyJoke',
      description:
        'Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke.',
      inputSchema: z.object({ topic: z.string() }),
      outputSchema: z.string(),
    },
    async (input) => {
      return `Why did the ${input.topic} cross the road?`;
    }
  );

  const namespacedPlugin: PluginProvider = {
    name: 'namespaced',
    initializer: async () => {},
  };
  registry.registerPluginProvider('namespaced', namespacedPlugin);

  defineTool(
    registry,
    {
      name: 'namespaced/add',
      description: 'add two numbers together',
      inputSchema: z.object({ a: z.number(), b: z.number() }),
      outputSchema: z.number(),
    },
    async ({ a, b }) => a + b
  );

  defineDynamicActionProvider(
    registry,
    {
      name: 'my-dap',
    },
    async () => {
      return {
        tool: [
          tool(
            {
              name: 'dapJokeTool',
              description: 'DAP joke tool',
              inputSchema: z.object({ topic: z.string() }),
              outputSchema: z.string(),
            },
            async ({ topic }) => `DAP joke about ${topic}`
          ),
        ],
      };
    }
  );

  const testCases = [
    {
      should: 'translate a string prompt correctly',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        prompt: 'Tell a joke about dogs.',
      },
      expectedOutput: {
        messages: [
          { role: 'user', content: [{ text: 'Tell a joke about dogs.' }] },
        ],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [],
        output: {},
      },
    },
    {
      should:
        'translate a string prompt correctly with tools referenced by their name',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        tools: ['tellAFunnyJoke'],
        prompt: 'Tell a joke about dogs.',
      },
      expectedOutput: {
        messages: [
          { role: 'user', content: [{ text: 'Tell a joke about dogs.' }] },
        ],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [
          {
            name: 'tellAFunnyJoke',
            description:
              'Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke.',
            outputSchema: {
              type: 'string',
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
            inputSchema: {
              type: 'object',
              properties: { topic: { type: 'string' } },
              required: ['topic'],
              additionalProperties: true,
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
            key: '/tool/tellAFunnyJoke',
          },
        ],
        output: {},
      },
    },
    {
      should: 'strip namespaces from tools when passing to the model',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        tools: ['namespaced/add'],
        prompt: 'Add 10 and 5.',
      },
      expectedOutput: {
        messages: [{ role: 'user', content: [{ text: 'Add 10 and 5.' }] }],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [
          {
            description: 'add two numbers together',
            inputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
              additionalProperties: true,
              properties: { a: { type: 'number' }, b: { type: 'number' } },
              required: ['a', 'b'],
              type: 'object',
            },
            name: 'add',
            outputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
              type: 'number',
            },
            key: '/tool/namespaced/add',
            metadata: {
              originalName: 'namespaced/add',
            },
          },
        ],
        output: {},
      },
    },
    {
      should:
        'translate a string prompt correctly with tools referenced by their action',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        tools: [tellAFunnyJoke],
        prompt: 'Tell a joke about dogs.',
      },
      expectedOutput: {
        messages: [
          { role: 'user', content: [{ text: 'Tell a joke about dogs.' }] },
        ],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [
          {
            name: 'tellAFunnyJoke',
            description:
              'Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke.',
            outputSchema: {
              type: 'string',
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
            inputSchema: {
              type: 'object',
              properties: { topic: { type: 'string' } },
              required: ['topic'],
              additionalProperties: true,
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
            key: '/tool/tellAFunnyJoke',
          },
        ],
        output: {},
      },
    },
    {
      should: 'translate a string prompt correctly with a DAP tool',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        tools: ['my-dap:tool/dapJokeTool'],
        prompt: 'Call DAP tool',
      },
      expectedOutput: {
        messages: [{ role: 'user', content: [{ text: 'Call DAP tool' }] }],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [
          {
            name: 'dapJokeTool',
            description: 'DAP joke tool',
            inputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
              type: 'object',
              properties: { topic: { type: 'string' } },
              required: ['topic'],
              additionalProperties: true,
            },
            outputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
              type: 'string',
            },
            key: '/dynamic-action-provider/my-dap:tool/dapJokeTool',
          },
        ],
        output: {},
      },
    },
    {
      should: 'translate a media prompt correctly',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        prompt: [
          { text: 'describe the following image:' },
          {
            media: {
              url: 'https://picsum.photos/200',
              contentType: 'image/jpeg',
            },
          },
        ],
      },
      expectedOutput: {
        messages: [
          {
            role: 'user',
            content: [
              { text: 'describe the following image:' },
              {
                media: {
                  url: 'https://picsum.photos/200',
                  contentType: 'image/jpeg',
                },
              },
            ],
          },
        ],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [],
        output: {},
      },
    },
    {
      should: 'translate a prompt with history correctly',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        messages: [
          { content: [{ text: 'hi' }], role: 'user' },
          { content: [{ text: 'how can I help you' }], role: 'model' },
        ],
        prompt: 'Tell a joke about dogs.',
      },
      expectedOutput: {
        messages: [
          { content: [{ text: 'hi' }], role: 'user' },
          { content: [{ text: 'how can I help you' }], role: 'model' },
          { role: 'user', content: [{ text: 'Tell a joke about dogs.' }] },
        ],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [],
        output: {},
      },
    },
    {
      should: 'pass context through to the model',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        prompt: 'Tell a joke with context.',
        docs: [{ content: [{ text: 'context here' }] }],
      },
      expectedOutput: {
        messages: [
          { content: [{ text: 'Tell a joke with context.' }], role: 'user' },
        ],
        config: undefined,
        docs: [{ content: [{ text: 'context here' }] }],
        resources: [],
        tools: [],
        output: {},
      },
    },
    {
      should:
        'throw a FAILED_PRECONDITION error if trying to resume without a model message',
      prompt: {
        messages: [{ role: 'system', content: [{ text: 'sys' }] }],
        resume: {
          respond: { toolResponse: { name: 'test', output: { foo: 'bar' } } },
        },
      },
      throws: 'FAILED_PRECONDITION',
    },
    {
      should:
        'throw a FAILED_PRECONDITION error if trying to resume a model message without toolRequests',
      prompt: {
        messages: [
          { role: 'user', content: [{ text: 'hi' }] },
          { role: 'model', content: [{ text: 'there' }] },
        ],
        resume: {
          respond: { toolResponse: { name: 'test', output: { foo: 'bar' } } },
        },
      },
      throws: 'FAILED_PRECONDITION',
    },
    {
      should: 'passes through output options',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        prompt: 'Tell a joke about dogs.',
        output: {
          constrained: true,
          format: 'banana',
        },
      },
      expectedOutput: {
        messages: [
          { role: 'user', content: [{ text: 'Tell a joke about dogs.' }] },
        ],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [],
        output: {
          constrained: true,
          format: 'banana',
        },
      },
    },
  ];
  for (const test of testCases) {
    it(test.should, async () => {
      if (test.throws) {
        await assert.rejects(
          async () => {
            await toGenerateRequest(registry, test.prompt as GenerateOptions);
          },
          { name: 'GenkitError', status: test.throws }
        );
      } else {
        const actualOutput = await toGenerateRequest(
          registry,
          test.prompt as GenerateOptions
        );
        assert.deepStrictEqual(actualOutput, test.expectedOutput);
      }
    });
  }
});

describe('toGenerateActionOptions', () => {
  const registry = new Registry();

  it('should return action options with undefined model', async () => {
    const options: GenerateOptions = {
      prompt: 'hello',
    };
    const actionOptions = await toGenerateActionOptions(registry, options);
    assert.strictEqual(actionOptions.model, undefined);
    assert.deepStrictEqual(actionOptions.messages, [
      { role: 'user', content: [{ text: 'hello' }] },
    ]);
  });
});

describe('generate', () => {
  let registry: Registry;
  var echoModel: ModelAction;

  beforeEach(() => {
    registry = new Registry();
    echoModel = defineModel(
      registry,
      {
        name: 'echoModel',
      },
      async (request) => {
        return {
          message: {
            role: 'model',
            content: [
              {
                text:
                  'Echo: ' +
                  request.messages
                    .map((m) => m.content.map((c) => c.text).join())
                    .join(),
              },
            ],
          },
          finishReason: 'stop',
        };
      }
    );
  });

  it('applies middleware', async () => {
    const wrapRequest: ModelMiddleware = async (req, next) => {
      return next({
        ...req,
        messages: [
          {
            role: 'user',
            content: [
              {
                text:
                  '(' +
                  req.messages
                    .map((m) => m.content.map((c) => c.text).join())
                    .join() +
                  ')',
              },
            ],
          },
        ],
      });
    };
    const wrapResponse: ModelMiddleware = async (req, next) => {
      const res = await next(req);
      return {
        message: {
          role: 'model',
          content: [
            {
              text: '[' + res.message!.content.map((c) => c.text).join() + ']',
            },
          ],
        },
        finishReason: res.finishReason,
      };
    };

    const response = await generate(registry, {
      prompt: 'banana',
      model: echoModel,
      use: [wrapRequest, wrapResponse],
    });
    const want = '[Echo: (banana)]';
    assert.deepStrictEqual(response.text, want);
  });
});

describe('generate', () => {
  let registry: Registry;
  beforeEach(() => {
    registry = new Registry();

    defineModel(
      registry,
      { name: 'echo', supports: { tools: true } },
      async (input) => ({
        message: input.messages[0],
        finishReason: 'stop',
      })
    );
  });

  it('should preserve the request in the returned response, enabling .messages', async () => {
    const response = await generate(registry, {
      model: 'echo',
      prompt: 'Testing messages',
    });
    assert.deepEqual(
      response.messages.map((m) => m.content[0].text),
      ['Testing messages', 'Testing messages']
    );
  });

  it('applies resources in the registry', async () => {
    defineResource(
      registry,
      { name: 'testResource', template: 'test://resource/{param}' },
      (input) => ({
        content: [{ text: 'resource' }, { text: input.uri }],
      })
    );

    const response = await generate(registry, {
      model: 'echo',
      prompt: [
        { text: 'some text' },
        { resource: { uri: 'test://resource/value' } },
      ],
    });
    assert.deepEqual(response.messages[0].content, [
      { text: 'some text' },
      {
        metadata: {
          resource: {
            template: 'test://resource/{param}',
            uri: 'test://resource/value',
          },
        },
        text: 'resource',
      },
      {
        metadata: {
          resource: {
            template: 'test://resource/{param}',
            uri: 'test://resource/value',
          },
        },
        text: 'test://resource/value',
      },
    ]);
  });

  it('throws when resource not found', async () => {
    const response = generate(registry, {
      model: 'echo',
      prompt: [{ text: 'some text' }, { resource: { uri: 'test://resource' } }],
    });
    await assert.rejects(response, {
      message:
        'NOT_FOUND: failed to find matching resource for test://resource',
    });
  });

  describe('generateStream', () => {
    it('should stream out chunks', async () => {
      const registry = new Registry();

      defineModel(
        registry,
        { name: 'echo-streaming', supports: { tools: true } },
        async (input, streamingCallback) => {
          streamingCallback!({ content: [{ text: 'hello, ' }] });
          streamingCallback!({ content: [{ text: 'world!' }] });
          return {
            message: input.messages[0],
            finishReason: 'stop',
          };
        }
      );

      const { response, stream } = generateStream(registry, {
        model: 'echo-streaming',
        prompt: 'Testing streaming',
      });

      const streamed: any[] = [];
      for await (const chunk of stream) {
        streamed.push(chunk.toJSON());
      }
      assert.deepStrictEqual(streamed, [
        {
          index: 0,
          role: 'model',
          content: [{ text: 'hello, ' }],
        },
        {
          index: 0,
          role: 'model',
          content: [{ text: 'world!' }],
        },
      ]);
      assert.deepEqual(
        (await response).messages.map((m) => m.content[0].text),
        ['Testing streaming', 'Testing streaming']
      );
    });

    it('should stream out chunks (v2 model)', async () => {
      const registry = new Registry();

      defineModel(
        registry,
        { apiVersion: 'v2', name: 'echo-streaming', supports: { tools: true } },
        async (input, { sendChunk }) => {
          sendChunk({ content: [{ text: 'hello, ' }] });
          sendChunk({ content: [{ text: 'world!' }] });
          return {
            message: input.messages[0],
            finishReason: 'stop',
          };
        }
      );

      const { response, stream } = generateStream(registry, {
        model: 'echo-streaming',
        prompt: 'Testing streaming',
      });

      const streamed: any[] = [];
      for await (const chunk of stream) {
        streamed.push(chunk.toJSON());
      }
      assert.deepStrictEqual(streamed, [
        {
          index: 0,
          role: 'model',
          content: [{ text: 'hello, ' }],
        },
        {
          index: 0,
          role: 'model',
          content: [{ text: 'world!' }],
        },
      ]);
      assert.deepEqual(
        (await response).messages.map((m) => m.content[0].text),
        ['Testing streaming', 'Testing streaming']
      );
    });
  });

  it('should use custom stepName parameter in tracing', async () => {
    const response = await generate(registry, {
      model: 'echo',
      prompt: 'Testing custom step name',
      stepName: 'test-generate-custom',
    });
    assert.deepEqual(
      response.messages.map((m) => m.content[0].text),
      ['Testing custom step name', 'Testing custom step name']
    );
  });

  it('should default to "generate" name when no stepName is provided', async () => {
    const response = await generate(registry, {
      model: 'echo',
      prompt: 'Testing default step name',
    });
    assert.deepEqual(
      response.messages.map((m) => m.content[0].text),
      ['Testing default step name', 'Testing default step name']
    );
  });

  it('handles multipart tool responses', async () => {
    defineTool(
      registry,
      {
        name: 'multiTool',
        description: 'a tool with multiple parts',
        multipart: true,
      },
      async () => {
        return {
          output: 'main output',
          content: [{ text: 'part 1' }],
          metadata: { custom: 'data' },
        };
      }
    );

    let requestCount = 0;
    defineModel(
      registry,
      { name: 'multi-tool-model', supports: { tools: true } },
      async (input) => {
        requestCount++;
        return {
          message: {
            role: 'model',
            content: [
              requestCount == 1
                ? {
                    toolRequest: {
                      name: 'multiTool',
                      input: {},
                    },
                  }
                : { text: 'done' },
            ],
          },
          finishReason: 'stop',
        };
      }
    );

    const response = await generate(registry, {
      model: 'multi-tool-model',
      prompt: 'go',
      tools: ['multiTool'],
    });
    assert.deepStrictEqual(response.messages, [
      {
        role: 'user',
        content: [
          {
            text: 'go',
          },
        ],
      },
      {
        role: 'model',
        content: [
          {
            toolRequest: {
              name: 'multiTool',
              input: {},
            },
          },
        ],
      },
      {
        role: 'tool',
        content: [
          {
            toolResponse: {
              name: 'multiTool',
              output: 'main output',
              content: [
                {
                  text: 'part 1',
                },
              ],
            },
            metadata: { custom: 'data' },
          },
        ],
      },
      {
        role: 'model',
        content: [
          {
            text: 'done',
          },
        ],
      },
    ]);
  });

  it('handles fallback tool responses', async () => {
    defineTool(
      registry,
      {
        name: 'fallbackTool',
        description: 'a tool with fallback output',
        multipart: true,
      },
      async () => {
        return {
          output: 'fallback output',
          content: [{ text: 'part 1' }],
        };
      }
    );

    let requestCount = 0;
    defineModel(
      registry,
      { name: 'fallback-tool-model', supports: { tools: true } },
      async (input) => {
        requestCount++;
        return {
          message: {
            role: 'model',
            content: [
              requestCount == 1
                ? {
                    toolRequest: {
                      name: 'fallbackTool',
                      input: {},
                    },
                  }
                : { text: 'done' },
            ],
          },
          finishReason: 'stop',
        };
      }
    );

    const response = await generate(registry, {
      model: 'fallback-tool-model',
      prompt: 'go',
      tools: ['fallbackTool'],
    });
    assert.deepStrictEqual(response.messages, [
      {
        role: 'user',
        content: [
          {
            text: 'go',
          },
        ],
      },
      {
        role: 'model',
        content: [
          {
            toolRequest: {
              name: 'fallbackTool',
              input: {},
            },
          },
        ],
      },
      {
        role: 'tool',
        content: [
          {
            toolResponse: {
              name: 'fallbackTool',
              output: 'fallback output',
              content: [
                {
                  text: 'part 1',
                },
              ],
            },
          },
        ],
      },
      {
        role: 'model',
        content: [
          {
            text: 'done',
          },
        ],
      },
    ]);
  });

  it('middleware can intercept streaming callback', async () => {
    const registry = new Registry();
    const echoModel = defineModel(
      registry,
      {
        apiVersion: 'v2',
        name: 'echoModel',
        supports: { tools: true },
      },
      async (_, { sendChunk }) => {
        if (sendChunk) {
          sendChunk({ content: [{ text: 'chunk1' }] });
          sendChunk({ content: [{ text: 'chunk2' }] });
        }
        return {
          message: {
            role: 'model',
            content: [{ text: 'done' }],
          },
          finishReason: 'stop',
        };
      }
    );

    const interceptMiddleware: ModelMiddlewareWithOptions = async (
      req,
      opts,
      next
    ) => {
      const originalOnChunk = opts!.onChunk;
      return next(req, {
        ...opts,
        onChunk: (chunk) => {
          if (originalOnChunk) {
            const text = chunk.content?.[0]?.text;
            originalOnChunk({
              ...chunk,
              content: [{ text: `intercepted: ${text}` }],
            });
          }
        },
      });
    };

    const { response, stream } = generateStream(registry, {
      model: echoModel,
      prompt: 'test',
      use: [interceptMiddleware],
    });

    const streamed: any[] = [];
    for await (const chunk of stream) {
      streamed.push(chunk.content[0].text);
    }

    assert.deepStrictEqual(streamed, [
      'intercepted: chunk1',
      'intercepted: chunk2',
    ]);
    await response;
  });

  it('middleware can modify context', async () => {
    const registry = new Registry();
    const checkContextModel = defineModel(
      registry,
      {
        apiVersion: 'v2',
        name: 'checkContextModel',
        supports: { context: true },
      },
      async (request, { context }) => {
        return {
          message: {
            role: 'model',
            content: [{ text: `Context: ${context?.myValue}` }],
          },
          finishReason: 'stop',
        };
      }
    );

    const contextMiddleware: ModelMiddlewareWithOptions = async (
      req,
      opts,
      next
    ) => {
      return next(req, {
        ...opts,
        context: {
          ...opts?.context,
          myValue: 'foo',
        },
      });
    };

    const response = await generate(registry, {
      model: checkContextModel,
      prompt: 'test',
      use: [contextMiddleware],
    });

    assert.strictEqual(response.text, 'Context: foo');
  });

  it('middleware can chain option modifications', async () => {
    const registry = new Registry();
    const checkContextModel = defineModel(
      registry,
      {
        apiVersion: 'v2',
        name: 'checkContextModel',
        supports: { context: true },
      },
      async (request, { context }) => {
        return {
          message: {
            role: 'model',
            content: [{ text: `Context: ${JSON.stringify(context)}` }],
          },
          finishReason: 'stop',
        };
      }
    );

    const middleware1: ModelMiddlewareWithOptions = async (req, opts, next) => {
      return next(req, {
        ...opts,
        context: {
          ...opts?.context,
          val: [...(opts?.context?.val ?? []), 'A'],
        },
      });
    };

    const middleware2: ModelMiddlewareWithOptions = async (req, opts, next) => {
      return next(req, {
        ...opts,
        context: {
          ...opts?.context,
          val: [...(opts?.context?.val ?? []), 'B'],
        },
      });
    };

    const response = await generate(registry, {
      model: checkContextModel,
      prompt: 'test',
      use: [middleware1, middleware2],
    });

    const context = JSON.parse(response.text.substring('Context: '.length));
    assert.deepStrictEqual(context.val, ['A', 'B']);
  });
});

describe('generate failures', () => {
  let registry: Registry;
  let modelCalls: number;

  beforeEach(() => {
    registry = new Registry();
    modelCalls = 0;
  });

  /**
   * Defines a model that asks for `toolName` on its first call and then runs
   * `then` on every later call, which either returns the final reply or
   * throws. Model responses carry a usage count so a partial's accounting can
   * be checked.
   */
  function defineToolLoopModel(
    toolName: string,
    then: (call: number) => GenerateResponseData
  ): ModelAction {
    return defineModel(
      registry,
      { name: 'loopModel', supports: { tools: true } },
      async () => {
        modelCalls++;
        if (modelCalls === 1) {
          return {
            message: {
              role: 'model',
              content: [{ toolRequest: { name: toolName, input: {}, ref: 'r1' } }],
            },
            finishReason: 'stop',
            usage: { inputTokens: modelCalls },
          };
        }
        return then(modelCalls);
      }
    );
  }

  function defineOkTool(name = 'okTool') {
    return defineTool(
      registry,
      { name, description: 'ok' },
      async () => 'ok'
    );
  }

  const roles = (r: GenerateResponse) => r.messages.map((m) => m.role);

  it('returns the completed rounds when a later model call fails', async () => {
    defineOkTool();
    defineToolLoopModel('okTool', () => {
      throw new GenkitError({
        status: 'UNAVAILABLE',
        message: 'model melted',
        detail: { provider: 'x' },
        responseMetadata: { retryAfterMs: 1000 },
      });
    });

    await assert.rejects(
      generate(registry, { model: 'loopModel', prompt: 'go', tools: ['okTool'] }),
      (e: any) => {
        assert.ok(e instanceof GenerationResponseError);
        assert.ok(!(e instanceof GenerationAbortedError));
        // The wrap is lossless: status, text, detail, retry metadata, cause.
        assert.strictEqual(e.status, 'UNAVAILABLE');
        assert.strictEqual(e.originalMessage, 'model melted');
        assert.strictEqual(e.detail.provider, 'x');
        assert.strictEqual(e.responseMetadata?.retryAfterMs, 1000);
        assert.ok(e.cause instanceof GenkitError);
        assert.strictEqual(e.cause.originalMessage, 'model melted');

        const partial: GenerateResponse = e.detail.response;
        assert.strictEqual(partial.finishReason, 'failed');
        assert.strictEqual(partial.finishMessage, 'model melted');
        assert.deepStrictEqual(partial.error, {
          status: 'UNAVAILABLE',
          message: 'model melted',
          details: { provider: 'x' },
        });
        assert.strictEqual(partial.message, undefined);
        // The seam: the caller's message and the completed tool round, nothing
        // from the failed call. A model call that failed has no accounting.
        assert.deepStrictEqual(roles(partial), ['user', 'model', 'tool']);
        assert.deepStrictEqual(
          partial.messages[2].content[0].toolResponse?.output,
          'ok'
        );
        assert.deepStrictEqual(partial.usage, {});
        return true;
      }
    );
    assert.strictEqual(modelCalls, 2);
  });

  it('drops the round a failed tool opened and reports the tool failure as INTERNAL', async () => {
    defineTool(registry, { name: 'badTool', description: 'bad' }, async () => {
      throw new Error('db password rejected');
    });
    defineToolLoopModel('badTool', () => ({
      message: { role: 'model', content: [{ text: 'unreachable' }] },
      finishReason: 'stop',
    }));

    await assert.rejects(
      generate(registry, { model: 'loopModel', prompt: 'go', tools: ['badTool'] }),
      (e: any) => {
        assert.ok(e instanceof GenerationResponseError);
        assert.strictEqual(e.status, 'INTERNAL');
        assert.strictEqual(
          e.originalMessage,
          'tool "badTool" failed: db password rejected'
        );
        // errors.Is/As in Go; the cause chain in JS reaches the tool's own error.
        assert.ok(e.cause instanceof GenkitError);
        assert.strictEqual((e.cause.cause as Error).message, 'db password rejected');

        const partial: GenerateResponse = e.detail.response;
        assert.strictEqual(partial.finishReason, 'failed');
        assert.strictEqual(partial.error?.status, 'INTERNAL');
        // The whole round goes, the model message that opened it included.
        assert.deepStrictEqual(roles(partial), ['user']);
        // The failing turn's accounting stays.
        assert.deepStrictEqual(partial.usage, { inputTokens: 1 });

        // A tool's arbitrary exception text is not written for a client; the
        // wire form carries the classification and the finish reason only.
        assert.deepStrictEqual(e.toJSON(), {
          status: 'INTERNAL',
          message: 'tool "badTool" failed',
          details: {
            finishReason: 'failed',
            finishMessage: 'tool "badTool" failed: db password rejected',
          },
        });
        return true;
      }
    );
    assert.strictEqual(modelCalls, 1);
  });

  it('keeps a tool GenkitError text on the wire and its status on the cause', async () => {
    defineTool(registry, { name: 'flaky', description: 'flaky' }, async () => {
      throw new GenkitError({ status: 'UNAVAILABLE', message: 'flaky tool failed' });
    });
    defineToolLoopModel('flaky', () => ({
      message: { role: 'model', content: [{ text: 'unreachable' }] },
      finishReason: 'stop',
    }));

    await assert.rejects(
      generate(registry, { model: 'loopModel', prompt: 'go', tools: ['flaky'] }),
      (e: any) => {
        assert.strictEqual(e.status, 'INTERNAL');
        assert.strictEqual(e.cause.cause.status, 'UNAVAILABLE');
        assert.strictEqual(e.toJSON().message, 'tool "flaky" failed: flaky tool failed');
        return true;
      }
    );
  });

  it('reports a failed tool as soon as it fails while its siblings finish detached', async () => {
    let siblingDone = false;
    defineTool(registry, { name: 'fast', description: 'fast' }, async () => {
      throw new Error('fast failed');
    });
    defineTool(registry, { name: 'slow', description: 'slow' }, async () => {
      await new Promise((r) => setTimeout(r, 30));
      siblingDone = true;
      return 'slow done';
    });
    defineModel(
      registry,
      { name: 'twoTools', supports: { tools: true } },
      async () => ({
        message: {
          role: 'model',
          content: [
            { toolRequest: { name: 'fast', input: {}, ref: 'a' } },
            { toolRequest: { name: 'slow', input: {}, ref: 'b' } },
          ],
        },
        finishReason: 'stop',
      })
    );

    await assert.rejects(
      generate(registry, { model: 'twoTools', prompt: 'go', tools: ['fast', 'slow'] }),
      (e: any) => {
        assert.strictEqual(e.originalMessage, 'tool "fast" failed: fast failed');
        assert.strictEqual(siblingDone, false);
        return true;
      }
    );
    await new Promise((r) => setTimeout(r, 50));
    assert.strictEqual(siblingDone, true);
  });

  it('stops at the turn limit with the round it refused to run dropped', async () => {
    defineOkTool();
    defineModel(
      registry,
      { name: 'alwaysTool', supports: { tools: true } },
      async () => {
        modelCalls++;
        return {
          message: {
            role: 'model',
            content: [
              { toolRequest: { name: 'okTool', input: {}, ref: `r${modelCalls}` } },
            ],
          },
          finishReason: 'stop',
          usage: { inputTokens: modelCalls },
        };
      }
    );

    await assert.rejects(
      generate(registry, {
        model: 'alwaysTool',
        prompt: 'go',
        tools: ['okTool'],
        maxTurns: 1,
      }),
      (e: any) => {
        assert.ok(e instanceof GenerationAbortedError);
        assert.strictEqual(e.status, 'ABORTED');
        assert.strictEqual(
          e.originalMessage,
          'Exceeded maximum tool call iterations (1)'
        );
        const partial: GenerateResponse = e.detail.response;
        assert.strictEqual(partial.finishReason, 'aborted');
        assert.strictEqual(partial.error?.status, 'ABORTED');
        assert.strictEqual(partial.message, undefined);
        // The first round completed; the second was refused and goes whole.
        assert.deepStrictEqual(roles(partial), ['user', 'model', 'tool']);
        assert.deepStrictEqual(partial.usage, { inputTokens: 2 });
        // The wire form carries no request.
        assert.strictEqual(e.toJSON().details.response, undefined);
        assert.strictEqual(JSON.stringify(e).includes('"messages"'), false);
        return true;
      }
    );
    assert.strictEqual(modelCalls, 2);
  });

  it('reports a signal that was already aborted before calling the model', async () => {
    defineToolLoopModel('okTool', () => ({
      message: { role: 'model', content: [{ text: 'unreachable' }] },
      finishReason: 'stop',
    }));
    const controller = new AbortController();
    controller.abort();

    await assert.rejects(
      generate(registry, {
        model: 'loopModel',
        prompt: 'go',
        abortSignal: controller.signal,
      }),
      (e: any) => {
        assert.ok(e instanceof GenerationAbortedError);
        assert.strictEqual(e.status, 'CANCELLED');
        const partial: GenerateResponse = e.detail.response;
        assert.strictEqual(partial.finishReason, 'aborted');
        assert.strictEqual(partial.error?.status, 'CANCELLED');
        assert.deepStrictEqual(roles(partial), ['user']);
        return true;
      }
    );
    assert.strictEqual(modelCalls, 0);
  });

  it('keeps a completed round when the caller aborts between turns', async () => {
    const controller = new AbortController();
    defineTool(registry, { name: 'stopper', description: 'stops' }, async () => {
      controller.abort();
      return 'done';
    });
    defineToolLoopModel('stopper', () => ({
      message: { role: 'model', content: [{ text: 'unreachable' }] },
      finishReason: 'stop',
    }));

    await assert.rejects(
      generate(registry, {
        model: 'loopModel',
        prompt: 'go',
        tools: ['stopper'],
        abortSignal: controller.signal,
      }),
      (e: any) => {
        assert.ok(e instanceof GenerationAbortedError);
        assert.strictEqual(e.status, 'CANCELLED');
        const partial: GenerateResponse = e.detail.response;
        assert.strictEqual(partial.finishReason, 'aborted');
        // The tool answered before the stop, so its round is a seam.
        assert.deepStrictEqual(roles(partial), ['user', 'model', 'tool']);
        return true;
      }
    );
    assert.strictEqual(modelCalls, 1);
  });

  it('classifies a tool that failed after the caller aborted as a stop, not a tool failure', async () => {
    const controller = new AbortController();
    defineTool(
      registry,
      { name: 'stopper', description: 'stops' },
      async (_, { abortSignal }) => {
        controller.abort();
        assert.strictEqual(abortSignal?.aborted, true); // the signal reaches the tool
        throw new Error('gave up');
      }
    );
    defineToolLoopModel('stopper', () => ({
      message: { role: 'model', content: [{ text: 'unreachable' }] },
      finishReason: 'stop',
    }));

    await assert.rejects(
      generate(registry, {
        model: 'loopModel',
        prompt: 'go',
        tools: ['stopper'],
        abortSignal: controller.signal,
      }),
      (e: any) => {
        assert.ok(e instanceof GenerationAbortedError);
        assert.strictEqual(e.status, 'CANCELLED');
        assert.strictEqual(e.originalMessage, 'tool "stopper" stopped: gave up');
        const partial: GenerateResponse = e.detail.response;
        assert.strictEqual(partial.finishReason, 'aborted');
        assert.deepStrictEqual(roles(partial), ['user']);
        return true;
      }
    );
  });

  it('reports a timed-out model call as aborted with DEADLINE_EXCEEDED', async () => {
    defineModel(registry, { name: 'slow' }, async () => {
      throw new DOMException('timed out', 'TimeoutError');
    });

    await assert.rejects(
      generate(registry, { model: 'slow', prompt: 'go' }),
      (e: any) => {
        assert.ok(e instanceof GenerationAbortedError);
        assert.strictEqual(e.status, 'DEADLINE_EXCEEDED');
        assert.strictEqual(e.detail.response.finishReason, 'aborted');
        return true;
      }
    );
  });

  it('treats a provider answering ABORTED as a failure, not a caller stop', async () => {
    defineModel(registry, { name: 'conflict' }, async () => {
      throw new GenkitError({ status: 'ABORTED', message: 'conflict' });
    });

    await assert.rejects(
      generate(registry, { model: 'conflict', prompt: 'go' }),
      (e: any) => {
        assert.ok(e instanceof GenerationResponseError);
        assert.ok(!(e instanceof GenerationAbortedError));
        assert.strictEqual(e.status, 'ABORTED');
        assert.strictEqual(e.detail.response.finishReason, 'failed');
        return true;
      }
    );
  });

  it('keeps the model message when structured output fails validation', async () => {
    defineModel(registry, { name: 'badJson' }, async () => ({
      message: { role: 'model', content: [{ text: '{"a": "not a number"}' }] },
      finishReason: 'stop',
    }));

    await assert.rejects(
      generate(registry, {
        model: 'badJson',
        prompt: 'go',
        output: { schema: z.object({ a: z.number() }) },
      }),
      (e: any) => {
        assert.ok(e instanceof GenerationResponseError);
        assert.strictEqual(e.status, 'INVALID_ARGUMENT');
        assert.ok(e.cause instanceof GenkitError);
        const partial: GenerateResponse = e.detail.response;
        // Not a loop stop: the model's own message and finish reason stay.
        assert.strictEqual(partial.finishReason, 'stop');
        assert.strictEqual(partial.text, '{"a": "not a number"}');
        assert.strictEqual(partial.error?.status, 'INVALID_ARGUMENT');
        assert.deepStrictEqual(roles(partial), ['user', 'model']);
        return true;
      }
    );
  });

  it('keeps an interrupted response when a restarted tool interrupts again', async () => {
    defineTool(
      registry,
      { name: 'confirm', description: 'confirm' },
      async () => {
        throw new ToolInterruptError({ again: true });
      }
    );
    defineToolLoopModel('confirm', () => ({
      message: { role: 'model', content: [{ text: 'unreachable' }] },
      finishReason: 'stop',
    }));

    await assert.rejects(
      generate(registry, {
        model: 'loopModel',
        tools: ['confirm'],
        messages: [
          { role: 'user', content: [{ text: 'hi' }] },
          {
            role: 'model',
            content: [
              {
                toolRequest: { name: 'confirm', input: {}, ref: 'c1' },
                metadata: { interrupt: true },
              },
            ],
          },
        ],
        resume: {
          restart: [{ toolRequest: { name: 'confirm', input: {}, ref: 'c1' } }],
        },
      }),
      (e: any) => {
        assert.ok(e instanceof GenerationResponseError);
        assert.strictEqual(e.status, 'FAILED_PRECONDITION');
        const partial: GenerateResponse = e.detail.response;
        assert.strictEqual(partial.finishReason, 'interrupted');
        assert.strictEqual(partial.error?.status, 'FAILED_PRECONDITION');
        // The revised model message is the tip, answered with `resume`.
        assert.deepStrictEqual(roles(partial), ['user', 'model']);
        assert.deepStrictEqual(partial.interrupts[0].metadata?.interrupt, {
          again: true,
        });
        assert.ok(e.detail.message);
        return true;
      }
    );
    assert.strictEqual(modelCalls, 0);
  });

  it('carries a partial for a resume the loop rejected', async () => {
    defineToolLoopModel('okTool', () => ({
      message: { role: 'model', content: [{ text: 'unreachable' }] },
      finishReason: 'stop',
    }));

    await assert.rejects(
      generate(registry, {
        model: 'loopModel',
        messages: [
          { role: 'user', content: [{ text: 'hi' }] },
          {
            role: 'model',
            content: [{ toolRequest: { name: 'other', input: {}, ref: 'o1' } }],
          },
        ],
        resume: {
          respond: [{ toolResponse: { name: 'nope', ref: 'x', output: 1 } }],
        },
      }),
      (e: any) => {
        assert.ok(e instanceof GenerationResponseError);
        assert.strictEqual(e.status, 'INVALID_ARGUMENT');
        assert.strictEqual(e.detail.response.finishReason, 'failed');
        assert.deepStrictEqual(roles(e.detail.response), ['user', 'model']);
        return true;
      }
    );
  });

  it('carries no partial for an error raised before the request resolved', async () => {
    defineModel(registry, { name: 'echo' }, async (input) => ({
      message: input.messages[0],
      finishReason: 'stop',
    }));

    await assert.rejects(
      generate(registry, {
        model: 'echo',
        prompt: [{ resource: { uri: 'test://missing' } }],
      }),
      (e: any) => {
        assert.strictEqual(e.status, 'NOT_FOUND');
        assert.strictEqual(e.detail?.response, undefined);
        return true;
      }
    );
  });

  it('restores the loop partial when a generate hook drops it', async () => {
    defineOkTool();
    defineToolLoopModel('okTool', () => {
      throw new GenkitError({ status: 'UNAVAILABLE', message: 'model melted' });
    });
    const dropping = generateMiddleware({ name: 'dropping' }, () => ({
      generate: async (envelope, ctx, next) => {
        try {
          return await next(envelope, ctx);
        } catch (e) {
          throw new Error('hook replaced');
        }
      },
    }));

    await assert.rejects(
      generate(registry, {
        model: 'loopModel',
        prompt: 'go',
        tools: ['okTool'],
        use: [dropping()],
      }),
      (e: any) => {
        assert.ok(e instanceof GenerationResponseError);
        // The hook's error is what the caller gets...
        assert.strictEqual(e.status, 'INTERNAL');
        assert.strictEqual(e.originalMessage, 'hook replaced');
        assert.strictEqual((e.cause as Error).message, 'hook replaced');
        // ...with the conversation the loop had completed, and the failure the
        // loop itself recorded, restored beside it.
        const partial: GenerateResponse = e.detail.response;
        assert.deepStrictEqual(roles(partial), ['user', 'model', 'tool']);
        assert.strictEqual(partial.error?.message, 'model melted');
        return true;
      }
    );
  });

  it('does not pair a recovered failure with a later hook error', async () => {
    defineModel(registry, { name: 'flakyModel' }, async () => {
      modelCalls++;
      if (modelCalls === 1) {
        throw new GenkitError({ status: 'UNAVAILABLE', message: 'model melted' });
      }
      return {
        message: { role: 'model', content: [{ text: 'recovered' }] },
        finishReason: 'stop',
      };
    });
    const retrying = generateMiddleware({ name: 'retrying' }, () => ({
      generate: async (envelope, ctx, next) => {
        try {
          await next(envelope, ctx);
        } catch (e) {
          // The retry succeeds; the hook then fails on its own terms.
          await next(envelope, ctx);
        }
        throw new Error('hook gave up');
      },
    }));

    await assert.rejects(
      generate(registry, { model: 'flakyModel', prompt: 'go', use: [retrying()] }),
      (e: any) => {
        assert.ok(e instanceof GenerationResponseError);
        const partial: GenerateResponse = e.detail.response;
        // Synthesized from the turn's request, not the stale first failure.
        assert.strictEqual(partial.error?.message, 'hook gave up');
        assert.strictEqual(partial.finishReason, 'failed');
        assert.deepStrictEqual(roles(partial), ['user']);
        return true;
      }
    );
    assert.strictEqual(modelCalls, 2);
  });

  it('rejects the streamed response with the same partial', async () => {
    defineOkTool();
    defineToolLoopModel('okTool', () => {
      throw new GenkitError({ status: 'UNAVAILABLE', message: 'model melted' });
    });

    const { stream, response } = generateStream(registry, {
      model: 'loopModel',
      prompt: 'go',
      tools: ['okTool'],
    });
    const chunks: string[] = [];
    await assert.rejects(
      (async () => {
        for await (const chunk of stream) {
          chunks.push(chunk.role ?? 'model');
        }
      })(),
      (e: any) => e instanceof GenerationResponseError
    );
    await assert.rejects(response, (e: any) => {
      assert.ok(e instanceof GenerationResponseError);
      assert.deepStrictEqual(roles(e.detail.response), ['user', 'model', 'tool']);
      return true;
    });
    // The completed round's tool message was streamed before the failure.
    assert.deepStrictEqual(chunks, ['tool']);
  });
});

describe('normalizeMiddleware', () => {
  it('handles legacy functional middleware by wrapping it', async () => {
    const registry = new Registry();
    const legacyMw = async (req: any, next: any) => {
      return next(req);
    };

    const refs = await normalizeMiddleware(registry, [legacyMw]);

    assert.strictEqual(refs.length, 1);
    assert.match(refs[0].name, /^dynamic-middleware-\d+-/);

    const registered = await registry.lookupValue<any>(
      'middleware',
      refs[0].name
    );
    assert.ok(registered);
  });

  it('handles MiddlewareRef objects created by calling middleware', async () => {
    const registry = new Registry();
    const myMw = generateMiddleware({ name: 'myMw' }, () => ({}));

    // Call it to get a MiddlewareRef
    const refs = await normalizeMiddleware(registry, [myMw()]);

    assert.strictEqual(refs.length, 1);
    assert.strictEqual(refs[0].name, 'myMw');

    const registered = await registry.lookupValue<any>('middleware', 'myMw');
    assert.ok(registered);
  });

  it('handles MiddlewareRef objects', async () => {
    const registry = new Registry();
    const myMw = generateMiddleware({ name: 'myMw' }, () => ({}));
    registry.registerValue('middleware', 'myMw', myMw);

    const refs = await normalizeMiddleware(registry, [{ name: 'myMw' }]);

    assert.strictEqual(refs.length, 1);
    assert.strictEqual(refs[0].name, 'myMw');
  });

  it('throws when uncalled middleware definition is passed as a function', async () => {
    const registry = new Registry();
    const myMw = generateMiddleware({ name: 'myMw' }, () => ({}));

    // Pass the definition function itself, which has .instantiate and .plugin
    await assert.rejects(
      async () => {
        await normalizeMiddleware(registry, [myMw as any]);
      },
      {
        name: 'GenkitError',
        status: 'INVALID_ARGUMENT',
      }
    );
  });
});
