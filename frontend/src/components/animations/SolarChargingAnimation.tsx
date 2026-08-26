/**
 * SolarChargingAnimation - GSAP 3-phase loop: charging → converting → selling
 *
 * This is a deliberately ABSTRACT illustration, not a data readout. It shows the
 * *concept* — light in, battery fills, energy converts to money, sells to grid —
 * and intentionally displays NO numeric SoC value: the real simulated SoC averages
 * ~23%, so printing a rising "80%" would be a false value claim. Only the gauge
 * filling and draining is shown. (Real per-hour SoC/SMP series are exposed on the
 * API for a future data-driven version; see soc_percent/smp_price_krw.)
 */
import { useEffect, useRef } from 'react';
import { Box } from '@chakra-ui/react';
import gsap from 'gsap';

// ── Geometry constants (340 × 500 viewBox) ──
const SUN_CX = 170;
const SUN_CY = 90;
const BAT_X  = 120;
const BAT_Y  = 272;
const BAT_W  = 100;
const BAT_H  = 140;
const FILL_X       = BAT_X + 2;              // 122
const FILL_W       = BAT_W - 4;              // 96
const FILL_BOTTOM  = BAT_Y + BAT_H - 2;      // 410
const FILL_MAX_H   = BAT_H - 4;              // 136
const FILL_80_H    = FILL_MAX_H * 0.8;       // 108.8
const FILL_80_Y    = FILL_BOTTOM - FILL_80_H; // ~301

const PARTICLE_Y_START = SUN_CY + 34; // 124 — just below the light source
const PARTICLE_Y_END   = BAT_Y - 2;  // 270 — just above battery top

// Particle stream: 10 emitters with per-index variation in horizontal spread,
// radius, speed and start delay so the flow reads as a continuous shimmer rather
// than three lockstep dots. Values are index-derived (deterministic, no random)
// so they stay stable across renders and are easy to reason about.
const PARTICLE_COUNT = 10;
const PARTICLES = Array.from({ length: PARTICLE_COUNT }, (_, i) => ({
  startX: SUN_CX + (i - (PARTICLE_COUNT - 1) / 2) * 7, // fan out under the light
  r: 3 + (i % 3),                                      // 3–5 px
  dur: 0.75 + (i % 4) * 0.13,                          // 0.75–1.14 s
  delay: (i % 5) * 0.34,                               // staggered starts
}));

const getBatteryFillColor = (socPct: number) => {
  if (socPct <= 10) return '#EF4444'; // red
  if (socPct <= 20) return '#FACC15'; // yellow
  return '#22C55E'; // green
};

export const SolarChargingAnimation = () => {
  const prefersReducedMotion =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const glowRef     = useRef<SVGCircleElement>(null);
  const particleRefs = useRef<(SVGCircleElement | null)[]>([]);
  const battFillRef = useRef<SVGRectElement>(null);
  const badgeRef    = useRef<SVGGElement>(null);
  const gridRef     = useRef<SVGGElement>(null);

  useEffect(() => {
    if (prefersReducedMotion) {
      // Static, meaningful frame: a half-filled gauge and a steady light source.
      if (battFillRef.current) {
        gsap.set(battFillRef.current, {
          attr: {
            height: FILL_MAX_H * 0.5,
            y: FILL_BOTTOM - FILL_MAX_H * 0.5,
            fill: getBatteryFillColor(50),
          },
        });
      }
      if (glowRef.current) gsap.set(glowRef.current, { opacity: 0.55 });
      return;
    }

    const particles = particleRefs.current;

    // ── Initial states ──
    gsap.set(glowRef.current, { opacity: 0.35, svgOrigin: `${SUN_CX} ${SUN_CY}` });
    gsap.set(particles, { opacity: 0 });
    gsap.set(battFillRef.current, { attr: { height: 0, y: FILL_BOTTOM, fill: getBatteryFillColor(0) } });
    gsap.set(badgeRef.current, { opacity: 0, x: 0, y: 0, scale: 1 });
    gsap.set(gridRef.current, { scale: 1, svgOrigin: '322 215' });

    const chargeProxy    = { value: 0 };
    const dischargeProxy = { value: 80 };

    const tl = gsap.timeline({ repeat: -1 });

    // ── Stage 1: CHARGING (0–4s) ──

    // Light source breathes as it "gathers" energy.
    tl.to(glowRef.current,
      { opacity: 0.75, scale: 1.08, duration: 1, ease: 'sine.inOut', yoyo: true, repeat: 3 },
      0,
    );

    // Particle stream — each emitter runs two passes to cover the 4s window,
    // converging from its fanned-out start x toward the battery center.
    particles.forEach((p, i) => {
      const cfg = PARTICLES[i];
      [cfg.delay, cfg.delay + 2.0].forEach((startT) => {
        tl.fromTo(p,
          { opacity: 0, attr: { cx: cfg.startX, cy: PARTICLE_Y_START } },
          { opacity: 0.85, attr: { cx: SUN_CX, cy: PARTICLE_Y_END }, duration: cfg.dur, ease: 'power1.in' },
          startT,
        );
        tl.to(p, { opacity: 0, duration: 0.22 }, startT + cfg.dur);
      });
    });

    tl.fromTo(battFillRef.current,
      { attr: { height: 0, y: FILL_BOTTOM } },
      { attr: { height: FILL_80_H, y: FILL_80_Y }, duration: 4, ease: 'power1.inOut' },
      0,
    );

    // Drive only the fill COLOR from the level (red→yellow→green). No number is
    // shown — the color is a qualitative cue, not a value claim.
    tl.fromTo(chargeProxy,
      { value: 0 },
      {
        value: 80, duration: 4, ease: 'power1.inOut',
        onUpdate: () => {
          const soc = Math.round(chargeProxy.value);
          if (battFillRef.current) gsap.set(battFillRef.current, { attr: { fill: getBatteryFillColor(soc) } });
        },
      },
      0,
    );

    // ── Stage 2: CONVERTING (4–5s) ──

    tl.fromTo(badgeRef.current,
      { opacity: 0, scale: 0.5, x: 0, y: 0, svgOrigin: `${SUN_CX} 210` },
      { opacity: 1, scale: 1.1, duration: 0.4, ease: 'back.out(1.7)' },
      4,
    );
    tl.to(badgeRef.current, { scale: 1, duration: 0.2 }, 4.4);

    // ── Stage 3: SELLING (5–7s) ──

    tl.set(badgeRef.current, { x: 0, y: 0 }, 4.99);
    tl.fromTo(badgeRef.current,
      { x: 0, y: 0, opacity: 1 },
      { x: 130, y: 10, opacity: 0, duration: 2, ease: 'power1.inOut', immediateRender: false },
      5,
    );
    tl.fromTo(
      gridRef.current,
      { scale: 1 },
      { scale: 1.08, duration: 0.18, ease: 'power2.out', yoyo: true, repeat: 1, immediateRender: false },
      6.72,
    );

    tl.fromTo(battFillRef.current,
      { attr: { height: FILL_80_H, y: FILL_80_Y } },
      { attr: { height: 0, y: FILL_BOTTOM }, duration: 2, ease: 'power1.inOut' },
      5,
    );

    tl.fromTo(dischargeProxy,
      { value: 80 },
      {
        value: 0, duration: 2, ease: 'power1.inOut',
        onUpdate: () => {
          const soc = Math.round(dischargeProxy.value);
          if (battFillRef.current) gsap.set(battFillRef.current, { attr: { fill: getBatteryFillColor(soc) } });
        },
      },
      5,
    );

    // ── Reset before repeat (7–7.8s) ──
    tl.set(badgeRef.current, { x: 0, y: 0, opacity: 0, scale: 1 }, 7);
    tl.set(gridRef.current, { scale: 1 }, 7);
    tl.set(battFillRef.current, { attr: { fill: getBatteryFillColor(0) } }, 7);
    tl.to({}, { duration: 0.8 }, 7);

    return () => {
      tl.kill();
    };
  }, [prefersReducedMotion]);

  return (
    <Box
      width="340px"
      height="500px"
      bg="spacex.darkGray"
      border="1px solid"
      borderColor="spacex.borderGray"
      display="flex"
      alignItems="center"
      justifyContent="center"
      position="relative"
      overflow="hidden"
    >
      <svg width="340" height="500" viewBox="0 0 340 500">
        <defs>
          {/* Soft radial light source — replaces the old clip-art sun */}
          <radialGradient id="lightGradX" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#FFF4CC" stopOpacity="0.95" />
            <stop offset="35%" stopColor="#FFD24D" stopOpacity="0.75" />
            <stop offset="70%" stopColor="#FFB020" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#FFB020" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="gridBoltGradX" x1="308" y1="199" x2="334" y2="229" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#93c5fd" />
            <stop offset="100%" stopColor="#38bdf8" />
          </linearGradient>
          <linearGradient id="gridPanelGradX" x1="300" y1="194" x2="344" y2="236" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#1e293b" />
            <stop offset="100%" stopColor="#0b1220" />
          </linearGradient>
        </defs>

        {/* Radial-gradient light source (animated opacity/scale via glowRef) */}
        <circle
          ref={glowRef}
          cx={SUN_CX} cy={SUN_CY} r="58"
          fill="url(#lightGradX)"
          opacity={0.35}
        />
        {/* Bright core of the light */}
        <circle cx={SUN_CX} cy={SUN_CY} r="10" fill="#FFF4CC" opacity={0.9} />

        {/* Energy path guide */}
        <line
          x1={SUN_CX} y1={PARTICLE_Y_START}
          x2={SUN_CX} y2={PARTICLE_Y_END}
          stroke="#FFD700" strokeWidth="1.5"
          strokeDasharray="5 5" opacity={0.15}
        />

        {/* Energy particles (10, index-varied) */}
        {PARTICLES.map((cfg, i) => (
          <circle
            key={i}
            ref={(el) => { particleRefs.current[i] = el; }}
            cx={cfg.startX} cy={PARTICLE_Y_START} r={cfg.r}
            fill="#FFD700" opacity={0}
          />
        ))}

        {/* Battery terminal nub */}
        <rect x={BAT_X + 28} y={BAT_Y - 10} width="44" height="10" rx="1"
          fill="#2d3748" stroke="white" strokeWidth="1.5" />

        {/* Battery container */}
        <rect x={BAT_X} y={BAT_Y} width={BAT_W} height={BAT_H}
          rx="2" fill="#111827" stroke="white" strokeWidth="2" />

        {/* Level graduation lines (unlabeled gauge ticks) */}
        {[25, 50, 75].map((pct) => (
          <line key={pct}
            x1={BAT_X}       y1={BAT_Y + BAT_H * (1 - pct / 100)}
            x2={BAT_X + BAT_W} y2={BAT_Y + BAT_H * (1 - pct / 100)}
            stroke="#4a5568" strokeWidth="1" opacity={0.4}
          />
        ))}

        {/* Clip path for battery fill */}
        <clipPath id="battClipX">
          <rect x={FILL_X} y={BAT_Y + 2} width={FILL_W} height={FILL_MAX_H} />
        </clipPath>

        {/* Battery fill rect (gauge only — no numeric label) */}
        <rect
          ref={battFillRef}
          x={FILL_X}
          y={FILL_BOTTOM}
          width={FILL_W}
          height={0}
          fill={getBatteryFillColor(0)}
          opacity={0.9}
          clipPath="url(#battClipX)"
        />

        {/* ₩ Conversion Badge */}
        <g ref={badgeRef} opacity={0}>
          <circle cx={SUN_CX} cy="210" r="22" fill="#facc15" stroke="#a16207" strokeWidth="2" />
          <circle cx={SUN_CX} cy="210" r="16" fill="none" stroke="#ca8a04" strokeWidth="1.1" opacity={0.85} />
          <text
            x={SUN_CX} y="210.5"
            textAnchor="middle" dominantBaseline="middle"
            fill="#7c2d12" fontSize="20" fontWeight="900"
            fontFamily="monospace"
          >
            {'₩'}
          </text>
        </g>

        {/* Upgraded power grid icon */}
        <g ref={gridRef}>
          <rect x="300" y="194" width="44" height="42" rx="6" fill="url(#gridPanelGradX)" stroke="#334155" strokeWidth="1.2" />
          <line x1="308" y1="204" x2="336" y2="204" stroke="#7dd3fc" strokeWidth="1.6" opacity={0.95} />
          <line x1="312" y1="204" x2="312" y2="228" stroke="#7dd3fc" strokeWidth="1.6" opacity={0.9} />
          <line x1="332" y1="204" x2="332" y2="228" stroke="#7dd3fc" strokeWidth="1.6" opacity={0.9} />
          <line x1="308" y1="228" x2="336" y2="228" stroke="#7dd3fc" strokeWidth="1.6" opacity={0.95} />
          <circle cx="312" cy="204" r="1.8" fill="#bae6fd" />
          <circle cx="332" cy="204" r="1.8" fill="#bae6fd" />
          <path
            d="M322 198 L315 212 H320 L318 227 L329 210 H324 Z"
            fill="url(#gridBoltGradX)"
            opacity={0.98}
          />
        </g>
      </svg>
    </Box>
  );
};

export default SolarChargingAnimation;
