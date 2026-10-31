// Pure-JS port of facenox/face-antispoof-onnx preprocessing (Apache-2.0).
// Works on {data: Uint8 RGB/RGBA, width, height, channels}.
(function (root) {
  function px(img, x, y, c) { return img.data[(y * img.width + x) * img.channels + c]; }

  // Bilinear resize to RGB float array (HWC)
  function resizeBilinear(img, ow, oh) {
    const out = new Float32Array(ow * oh * 3);
    const sx = img.width / ow, sy = img.height / oh;
    for (let y = 0; y < oh; y++) {
      let fy = (y + 0.5) * sy - 0.5; if (fy < 0) fy = 0;
      const y0 = Math.min(Math.floor(fy), img.height - 1), y1 = Math.min(y0 + 1, img.height - 1), wy = fy - y0;
      for (let x = 0; x < ow; x++) {
        let fx = (x + 0.5) * sx - 0.5; if (fx < 0) fx = 0;
        const x0 = Math.min(Math.floor(fx), img.width - 1), x1 = Math.min(x0 + 1, img.width - 1), wx = fx - x0;
        for (let c = 0; c < 3; c++) {
          const v = (px(img, x0, y0, c) * (1 - wx) + px(img, x1, y0, c) * wx) * (1 - wy) +
                    (px(img, x0, y1, c) * (1 - wx) + px(img, x1, y1, c) * wx) * wy;
          out[(y * ow + x) * 3 + c] = v;
        }
      }
    }
    return out;
  }
  // Area-average resize (downscale), like cv2.INTER_AREA
  function resizeArea(img, ow, oh) {
    const out = new Float32Array(ow * oh * 3);
    const sx = img.width / ow, sy = img.height / oh;
    for (let y = 0; y < oh; y++) {
      const ya = y * sy, yb = (y + 1) * sy;
      for (let x = 0; x < ow; x++) {
        const xa = x * sx, xb = (x + 1) * sx;
        let s0 = 0, s1 = 0, s2 = 0, wt = 0;
        for (let yy = Math.floor(ya); yy < Math.ceil(yb) && yy < img.height; yy++) {
          const wy = Math.min(yb, yy + 1) - Math.max(ya, yy);
          for (let xx = Math.floor(xa); xx < Math.ceil(xb) && xx < img.width; xx++) {
            const w = wy * (Math.min(xb, xx + 1) - Math.max(xa, xx));
            const i = (yy * img.width + xx) * img.channels;
            s0 += img.data[i] * w; s1 += img.data[i + 1] * w; s2 += img.data[i + 2] * w; wt += w;
          }
        }
        const o = (y * ow + x) * 3; out[o] = s0 / wt; out[o + 1] = s1 / wt; out[o + 2] = s2 / wt;
      }
    }
    return out;
  }
  function resize(img, ow, oh) {
    return (ow < img.width || oh < img.height) ? resizeArea(img, ow, oh) : resizeBilinear(img, ow, oh);
  }

  // ---------- YuNet detector ----------
  const DET = 640;
  function detectorInput(img) {
    const scale = DET / Math.max(img.width, img.height);
    const nw = Math.max(1, Math.round(img.width * scale)), nh = Math.max(1, Math.round(img.height * scale));
    const r = resize(img, nw, nh);
    const t = new Float32Array(3 * DET * DET); // zero padded right/bottom
    for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) for (let c = 0; c < 3; c++)
      t[c * DET * DET + y * DET + x] = r[(y * nw + x) * 3 + c];
    return { tensor: t, scale };
  }
  function iou(a, b) {
    const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    return inter / (a[2] * a[3] + b[2] * b[3] - inter);
  }
  function decodeDetections(outs, scale, scoreThr = 0.8, nmsThr = 0.3) {
    const cands = [];
    for (const s of [8, 16, 32]) {
      const cols = DET / s, rows = DET / s;
      const cls = outs['cls_' + s], obj = outs['obj_' + s], bb = outs['bbox_' + s];
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        const cs = Math.min(Math.max(cls[i], 0), 1), os = Math.min(Math.max(obj[i], 0), 1);
        const score = Math.sqrt(cs * os);
        if (score < scoreThr) continue;
        const cx = (c + bb[i * 4]) * s, cy = (r + bb[i * 4 + 1]) * s;
        const w = Math.exp(bb[i * 4 + 2]) * s, h = Math.exp(bb[i * 4 + 3]) * s;
        cands.push({ box: [(cx - w / 2) / scale, (cy - h / 2) / scale, w / scale, h / scale], score });
      }
    }
    cands.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const d of cands) if (keep.every(k => iou(k.box, d.box) <= nmsThr)) keep.push(d);
    return keep;
  }

  // ---------- Liveness crop + preprocess ----------
  function reflect101(i, n) {
    if (n === 1) return 0;
    while (i < 0 || i >= n) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i; }
    return i;
  }
  function cropFace(img, box, expansion = 1.5) {
    let [x, y, w, h] = box;
    const maxDim = Math.max(w, h), cx = x + w / 2, cy = y + h / 2;
    const size = Math.trunc(maxDim * expansion);
    const x0 = Math.trunc(cx - maxDim * expansion / 2), y0 = Math.trunc(cy - maxDim * expansion / 2);
    // region actually inside the image (python: img[cy1:cy2, cx1:cx2] then reflect pad)
    const cx1 = Math.max(0, x0), cy1 = Math.max(0, y0);
    const cx2 = Math.min(img.width, x0 + size), cy2 = Math.min(img.height, y0 + size);
    const iw = cx2 - cx1, ih = cy2 - cy1;
    const data = new Uint8ClampedArray(size * size * 3);
    for (let yy = 0; yy < size; yy++) {
      const sy = cy1 + reflect101(yy - (cy1 - y0), ih);
      for (let xx = 0; xx < size; xx++) {
        const sx = cx1 + reflect101(xx - (cx1 - x0), iw);
        for (let c = 0; c < 3; c++) data[(yy * size + xx) * 3 + c] = px(img, sx, sy, c);
      }
    }
    return { data, width: size, height: size, channels: 3 };
  }
  function livenessInput(face, S = 128) {
    const ratio = S / Math.max(face.width, face.height);
    const nw = Math.trunc(face.width * ratio), nh = Math.trunc(face.height * ratio);
    const r = resize(face, nw, nh);
    const top = Math.floor((S - nh) / 2), left = Math.floor((S - nw) / 2);
    const t = new Float32Array(3 * S * S);
    for (let y = 0; y < S; y++) {
      const sy = reflect101(y - top, nh);
      for (let x = 0; x < S; x++) {
        const sx = reflect101(x - left, nw);
        for (let c = 0; c < 3; c++) t[c * S * S + y * S + x] = Math.round(Math.min(255, Math.max(0, r[(sy * nw + sx) * 3 + c]))) / 255;
      }
    }
    return t;
  }
  function verdict(logits, threshold = 0.5) {
    const diff = logits[0] - logits[1];
    const p = Math.min(1 - 1e-6, Math.max(1e-6, threshold));
    const thr = Math.log(p / (1 - p));
    return { isReal: diff >= thr, diff, pReal: 1 / (1 + Math.exp(-diff)) };
  }
  const api = { detectorInput, decodeDetections, cropFace, livenessInput, verdict, DET };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.FAS = api;
})(typeof self !== 'undefined' ? self : this);
