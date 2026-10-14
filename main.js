// CS405 · Lab 2 — three squares in a hierarchy

import {
  identity,
  multiply as mul,
  translate,
  rotate,
  scale,
  pad3x3,
  invert,
  apply,
  selfTest
} from './mat3.js';

selfTest();

const canvas = document.querySelector('canvas');

if (!navigator.gpu) {
  throw new Error('WebGPU is not available in this browser.');
}

const adapter = await navigator.gpu.requestAdapter();

if (!adapter) {
  throw new Error('No WebGPU adapter — check webgpureport.org.');
}

const device = await adapter.requestDevice();
const ctx = canvas.getContext('webgpu');
const format = navigator.gpu.getPreferredCanvasFormat();

ctx.configure({
  device,
  format,
  alphaMode: 'opaque'
});

const module = device.createShaderModule({
  code: /* wgsl */`
struct U {
  m: mat3x3f,
  colour: vec4f
};

@group(0) @binding(0) var<uniform> u: U;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var q = array<vec2f, 6>(
    vec2f(-0.5, -0.5),
    vec2f( 0.5, -0.5),
    vec2f( 0.5,  0.5),

    vec2f(-0.5, -0.5),
    vec2f( 0.5,  0.5),
    vec2f(-0.5,  0.5)
  );

  let p = u.m * vec3f(q[i], 1.0);

  return vec4f(p.xy, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4f {
  return u.colour;
}
`
});

const pipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: {
    module,
    entryPoint: 'vs'
  },
  fragment: {
    module,
    entryPoint: 'fs',
    targets: [{ format }]
  }
});


const objects = [
  { colour: [0.55, 0.53, 0.5, 1] },
  { colour: [0.12, 0.37, 0.55, 1] },
  { colour: [0.71, 0.27, 0.11, 1] }
].map(o => {
  const buf = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: buf }
      }
    ]
  });

  return { ...o, buf, bind };
});

function resize() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const r = canvas.getBoundingClientRect();

  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
}

window.addEventListener('resize', resize);
resize();


let mouse = null;
let reach = false;
let hovered = -1;

canvas.addEventListener('pointermove', e => {
  const r = canvas.getBoundingClientRect();

  mouse = [
    2 * (e.clientX - r.left) / r.width - 1,
    1 - 2 * (e.clientY - r.top) / r.height
  ];
});

canvas.addEventListener('pointerleave', () => {
  mouse = null;
});

window.addEventListener('keydown', e => {
  if (e.key === 'i' || e.key === 'I') {
    reach = !reach;
  }
});


const L1 = 0.30;
const L2 = 0.22;

function ik2(x, y) {
  const rawC2 =
    (x * x + y * y - L1 * L1 - L2 * L2) /
    (2 * L1 * L2);

  const c2 = Math.max(-1, Math.min(1, rawC2));
  const theta2 = -Math.acos(c2);

  const theta1 =
    Math.atan2(y, x) -
    Math.atan2(
      L2 * Math.sin(theta2),
      L1 + L2 * Math.cos(theta2)
    );

  return [theta1 - Math.PI / 2, theta2];
}

const t0 = performance.now();

function frame() {
  const t = (performance.now() - t0) / 1000;
  const aspect = canvas.width / canvas.height;

  const a0 = 0.35 * Math.sin(t * 0.7);
  let a1 = 0.8 * Math.sin(t * 1.1);
  let a2 = 1.1 * Math.sin(t * 1.7);

  const box = (w, h) =>
    mul(translate(0, h / 2), scale(w, h));

  const aspectFix = scale(1 / aspect, 1);

  
  if (reach && mouse) {
    const shoulder = mul(
      translate(0, -0.55),
      mul(rotate(a0), translate(0, 0.30))
    );

    const shoulderInverse = invert(mul(aspectFix, shoulder));

    if (shoulderInverse) {
      const [x, y] = apply(
        shoulderInverse,
        mouse[0],
        mouse[1]
      );

      [a1, a2] = ik2(x, y);
    }
  }

  
  const baseLocal = mul(
    translate(0, -0.55),
    rotate(a0)
  );

  const armLocal = mul(
  translate(0, 0.30),
  rotate(a1)
  );

  const handLocal = mul(
    translate(0, 0.30),
    rotate(a2)
  );

  
  const baseWorld = baseLocal;
  const armWorld = mul(baseWorld, armLocal);
  const handWorld = mul(armWorld, handLocal);

  const worlds = [
    mul(aspectFix, mul(baseWorld, box(0.5, 0.18))),
    mul(aspectFix, mul(armWorld, box(0.22, 0.34))),
    mul(aspectFix, mul(handWorld, box(0.16, 0.22)))
  ];

  
  hovered = -1;

  if (mouse) {
    worlds.forEach((m, i) => {
      const inv = invert(m);

      if (!inv) return;

      const [x, y] = apply(inv, mouse[0], mouse[1]);

      if (Math.abs(x) <= 0.5 && Math.abs(y) <= 0.5) {
        hovered = i;
      }
    });
  }

  worlds.forEach((m, i) => {
    const c = objects[i].colour;

    const col = i === hovered
      ? [
          0.5 + 0.5 * c[0],
          0.5 + 0.5 * c[1],
          0.5 + 0.5 * c[2],
          1
        ]
      : c;

    device.queue.writeBuffer(
      objects[i].buf,
      0,
      pad3x3(m)
    );

    device.queue.writeBuffer(
      objects[i].buf,
      48,
      new Float32Array(col)
    );
  });

  const enc = device.createCommandEncoder();

  const pass = enc.beginRenderPass({
    colorAttachments: [
      {
        view: ctx.getCurrentTexture().createView(),
        clearValue: {
          r: 0.06,
          g: 0.07,
          b: 0.09,
          a: 1
        },
        loadOp: 'clear',
        storeOp: 'store'
      }
    ]
  });

  pass.setPipeline(pipeline);

  objects.forEach(o => {
    pass.setBindGroup(0, o.bind);
    pass.draw(6);
  });

  pass.end();
  device.queue.submit([enc.finish()]);

  requestAnimationFrame(frame);
}

frame();