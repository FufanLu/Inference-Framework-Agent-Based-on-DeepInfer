import type { ToolSpec } from "@/api/protocols";

/**
 * Demo tools that run in the browser. None of them reaches any outside
 * service: the weather tool returns made-up data derived from the city name
 * and says so in its result. The specs are model-facing, so they stay in one
 * language whatever the UI locale; UI labels come from the dictionaries.
 */

export interface DemoTool {
  spec: ToolSpec;
  run: (args: Record<string, unknown>) => unknown;
}

const CITIES = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou", "Chengdu"] as const;

export const WEATHER_TOOL: ToolSpec = {
  name: "get_weather",
  description: "Get the current weather for a city (demo tool: the data is generated in the browser, not a real forecast)",
  parameters: {
    type: "object",
    properties: {
      city: { type: "string", description: "City name", enum: [...CITIES] },
    },
    required: ["city"],
  },
};

const CALCULATOR_TOOL: ToolSpec = {
  name: "calculator",
  description: "Evaluate an arithmetic expression. Separate numbers and + - * / with spaces, e.g. 12.5 * 4 + 3",
  parameters: {
    type: "object",
    properties: {
      expression: {
        type: "string",
        description: "Arithmetic expression",
        pattern: "^\\d{1,4}(\\.\\d{1,2})? [+\\-*/] \\d{1,4}( [+\\-*/] \\d{1,3})?$",
      },
    },
    required: ["expression"],
  },
};

const CONVERSIONS = {
  "km->mi": { label: "kilometres → miles", f: (v: number) => v / 1.609344, unit: "mi" },
  "mi->km": { label: "miles → kilometres", f: (v: number) => v * 1.609344, unit: "km" },
  "m->ft": { label: "metres → feet", f: (v: number) => v / 0.3048, unit: "ft" },
  "ft->m": { label: "feet → metres", f: (v: number) => v * 0.3048, unit: "m" },
  "kg->lb": { label: "kilograms → pounds", f: (v: number) => v / 0.45359237, unit: "lb" },
  "lb->kg": { label: "pounds → kilograms", f: (v: number) => v * 0.45359237, unit: "kg" },
  "C->F": { label: "Celsius → Fahrenheit", f: (v: number) => (v * 9) / 5 + 32, unit: "°F" },
  "F->C": { label: "Fahrenheit → Celsius", f: (v: number) => ((v - 32) * 5) / 9, unit: "°C" },
} as const;

type Conversion = keyof typeof CONVERSIONS;

const CONVERT_TOOL: ToolSpec = {
  name: "convert_units",
  description: "Convert units of length, mass or temperature",
  parameters: {
    type: "object",
    properties: {
      value: { type: "number", description: "Value to convert", minimum: -1000, maximum: 100000 },
      conversion: { type: "string", description: "Direction of the conversion", enum: Object.keys(CONVERSIONS) },
    },
    required: ["value", "conversion"],
  },
};

// ------------------------------------------------------------ calculator

/** A small recursive-descent evaluator: + - * / % ^, parentheses, a few functions. No eval. */
export function evaluate(expression: string): number {
  const source = expression.replace(/×/g, "*").replace(/÷/g, "/");
  let pos = 0;
  const peek = () => source[pos];
  const skip = () => {
    while (pos < source.length && /\s/.test(source[pos] ?? "")) pos += 1;
  };
  const fail = (message: string): never => {
    throw new Error(`${message} (at ${pos})`);
  };
  const FUNCTIONS: Record<string, (...args: number[]) => number> = {
    sqrt: Math.sqrt,
    abs: Math.abs,
    round: Math.round,
    floor: Math.floor,
    ceil: Math.ceil,
    sin: Math.sin,
    cos: Math.cos,
    tan: Math.tan,
    ln: Math.log,
    log: Math.log10,
    min: Math.min,
    max: Math.max,
    pow: Math.pow,
  };
  const CONSTANTS: Record<string, number> = { pi: Math.PI, e: Math.E };

  function primary(): number {
    skip();
    const ch = peek();
    if (ch === "(") {
      pos += 1;
      const value = sum();
      skip();
      if (peek() !== ")") fail("missing closing parenthesis");
      pos += 1;
      return value;
    }
    if (ch === "-" || ch === "+") {
      pos += 1;
      const value = power();
      return ch === "-" ? -value : value;
    }
    const number = /^(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?/.exec(source.slice(pos));
    if (number) {
      pos += number[0].length;
      return Number(number[0]);
    }
    const name = /^[a-zA-Z]+/.exec(source.slice(pos));
    if (name) {
      const id = name[0].toLowerCase();
      pos += name[0].length;
      if (id in CONSTANTS) return CONSTANTS[id] ?? 0;
      const fn = FUNCTIONS[id];
      if (!fn) fail(`unknown function ${id}`);
      skip();
      if (peek() !== "(") fail(`${id} needs parentheses`);
      pos += 1;
      const args = [sum()];
      skip();
      while (peek() === ",") {
        pos += 1;
        args.push(sum());
        skip();
      }
      if (peek() !== ")") fail("missing closing parenthesis");
      pos += 1;
      return (fn as (...a: number[]) => number)(...args);
    }
    return fail(ch === undefined ? "unexpected end of expression" : `unexpected character "${ch}"`);
  }
  function power(): number {
    const base = primary();
    skip();
    if (peek() === "^") {
      pos += 1;
      return base ** power();
    }
    return base;
  }
  function product(): number {
    let value = power();
    for (;;) {
      skip();
      const op = peek();
      if (op !== "*" && op !== "/" && op !== "%") return value;
      pos += 1;
      const right = power();
      value = op === "*" ? value * right : op === "/" ? value / right : value % right;
    }
  }
  function sum(): number {
    let value = product();
    for (;;) {
      skip();
      const op = peek();
      if (op !== "+" && op !== "-") return value;
      pos += 1;
      const right = product();
      value = op === "+" ? value + right : value - right;
    }
  }

  if (!source.trim()) fail("empty expression");
  const result = sum();
  skip();
  if (pos < source.length) fail(`unexpected trailing input "${source.slice(pos)}"`);
  if (!Number.isFinite(result)) throw new Error("the result is not finite (division by zero?)");
  return result;
}

// --------------------------------------------------------------- weather

function hash(text: string): number {
  let h = 2166136261;
  for (const ch of text) {
    h ^= ch.codePointAt(0) ?? 0;
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function demoWeather(city: string) {
  const h = hash(city);
  const conditions = ["sunny", "partly cloudy", "overcast", "light rain", "thunderstorms", "fog"];
  return {
    city,
    condition: conditions[h % conditions.length],
    temperature_c: Math.round(((h >>> 8) % 300) / 10 - 2),
    humidity_pct: 30 + ((h >>> 16) % 60),
    wind_kph: (h >>> 4) % 30,
    note: "demo data generated in the browser from the city name; not a real forecast",
  };
}

// ---------------------------------------------------------------- tools

export const DEMO_TOOLS: DemoTool[] = [
  {
    spec: WEATHER_TOOL,
    run: (args) => {
      const city = typeof args.city === "string" && args.city.trim() ? args.city.trim() : null;
      if (!city) return { error: "missing city argument" };
      return demoWeather(city);
    },
  },
  {
    spec: CALCULATOR_TOOL,
    run: (args) => {
      const expression = typeof args.expression === "string" ? args.expression : "";
      try {
        return { expression, result: Number(evaluate(expression).toPrecision(12)) };
      } catch (error) {
        return { expression, error: error instanceof Error ? error.message : String(error) };
      }
    },
  },
  {
    spec: CONVERT_TOOL,
    run: (args) => {
      const value = typeof args.value === "number" ? args.value : Number(args.value);
      const conversion = String(args.conversion ?? "") as Conversion;
      const rule = CONVERSIONS[conversion];
      if (!rule) return { error: `unsupported conversion ${conversion}; one of ${Object.keys(CONVERSIONS).join(", ")}` };
      if (!Number.isFinite(value)) return { error: "value is not a number" };
      return { value, conversion, label: rule.label, result: Number(rule.f(value).toPrecision(8)), unit: rule.unit };
    },
  },
];

export function findTool(name: string): DemoTool | undefined {
  return DEMO_TOOLS.find((tool) => tool.spec.name === name);
}
