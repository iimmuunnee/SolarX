/**
 * SolarChargingAnimation - GSAP 3-phase loop: charging → converting → selling
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

const PARTICLE_Y_START = SUN_CY + 34; // 124 — just below sun edge
const PARTICLE_Y_END   = BAT_Y - 2;  // 270 — just above battery top

const SUN_RAYS = Array.from({ length: 8 }, (_, i) => i);

const getBatteryFillColor = (socPct: number) => {
  if (socPct <= 10) return '#EF4444'; // red
  if (socPct <= 20) return '#FACC15'; // yellow
  return '#22C55E'; // green
};

export const SolarChargingAnimation = () => {
  const prefersReducedMotion =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const glowRef    = useRef<SVGCircleElement>(null);
  const p0Ref      = useRef<SVGCircleElement>(null);
  const p1Ref      = useRef<SVGCircleElement>(null);
  const p2Ref      = useRef<SVGCircleElement>(null);
  const battFillRef = useRef<SVGRectElement>(null);
  const battPctRef  = useRef<SVGTextElement>(null);
  const badgeRef    = useRef<SVGGElement>(null);
  const gridRef     = useRef<SVGGElement>(null);

  useEffect(() => {
    if (prefersReducedMotion) {
      if (battFillRef.current) {
        gsap.set(battFillRef.current, {
          attr: {
            height: FILL_MAX_H * 0.5,
            y: FILL_BOTTOM - FILL_MAX_H * 0.5,
            fill: getBatteryFillColor(50),
          },
        });
      }
      if (battPctRef.current) battPctRef.current.textContent = '50%';
      return;
    }

    const particles = [p0Ref.current, p1Ref.current, p2Ref.current];

    // ── Initial states ──
    gsap.set(glowRef.current, { opacity: 0 });
    gsap.set(particles, { opacity: 0, attr: { cy: PARTICLE_Y_START } });
    gsap.set(battFillRef.current, { attr: { height: 0, y: FILL_BOTTOM, fill: getBatteryFillColor(0) } });
    gsap.set(badgeRef.current, { opacity: 0, x: 0, y: 0, scale: 1 });
    gsap.set(gridRef.current, { scale: 1, svgOrigin: '322 215' });
    if (battPctRef.current) battPctRef.current.textContent = '0%';

    const chargeProxy    = { value: 0 };
    const dischargeProxy = { value: 80 };

    const tl = gsap.timeline({ repeat: -1 });

    // ── Stage 1: CHARGING (0–4s) ──

    tl.fromTo(glowRef.current,
      { opacity: 0 },
      { opacity: 0.4, duration: 1, ease: 'sine.inOut', yoyo: true, repeat: 3 },
      0,
    );

    const MOVE_DUR    = 0.9;
    const FADE_DUR    = 0.25;
    const STAGGER     = 0.4;
    const PASS2_START = 1.9;

    particles.forEach((p, i) => {
      const s1 = i * STAGGER;
      const s2 = PASS2_START + i * STAGGER;

      tl.fromTo(p,
        { opacity: 0, attr: { cy: PARTICLE_Y_START } },
        { opacity: 0.9, attr: { cy: PARTICLE_Y_END }, duration: MOVE_DUR, ease: 'power1.in' },
        s1,
      );
      tl.to(p, { opacity: 0, duration: FADE_DUR }, s1 + MOVE_DUR);

      tl.fromTo(p,
        { opacity: 0, attr: { cy: PARTICLE_Y_START } },
        { opacity: 0.9, attr: { cy: PARTICLE_Y_END }, duration: MOVE_DUR, ease: 'power1.in' },
        s2,
      );
      tl.to(p, { opacity: 0, duration: FADE_DUR }, s2 + MOVE_DUR);
    });

    tl.fromTo(battFillRef.current,
      { attr: { height: 0, y: FILL_BOTTOM } },
      { attr: { height: FILL_80_H, y: FILL_80_Y }, duration: 4, ease: 'power1.inOut' },
      0,
    );

    tl.fromTo(chargeProxy,
      { value: 0 },
      {
        value: 80, duration: 4, ease: 'power1.inOut',
        onUpdate: () => {
          const soc = Math.round(chargeProxy.value);
          if (battPctRef.current) battPctRef.current.textContent = `${soc}%`;
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
          if (battPctRef.current) battPctRef.current.textContent = `${soc}%`;
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
          <filter id="sunGlowX">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <linearGradient id="gridBoltGradX" x1="308" y1="199" x2="334" y2="229" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#93c5fd" />
            <stop offset="100%" stopColor="#38bdf8" />
          </linearGradient>
          <linearGradient id="gridPanelGradX" x1="300" y1="194" x2="344" y2="236" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#1e293b" />
            <stop offset="100%" stopColor="#0b1220" />
          </linearGradient>
        </defs>

        {/* Glow ring */}
        <circle
          ref={glowRef}
          cx={SUN_CX} cy={SUN_CY} r="48"
          fill="none" stroke="#FFD700" strokeWidth="2.5"
          opacity={0}
        />

        {/* Sun rays (static) */}
        {SUN_RAYS.map((i) => {
          const angle = (i * 45 * Math.PI) / 180;
          const x1 = SUN_CX + 30 * Math.cos(angle);
          const y1 = SUN_CY + 30 * Math.sin(angle);
          const x2 = SUN_CX + 46 * Math.cos(angle);
          const y2 = SUN_CY + 46 * Math.sin(angle);
          return (
            <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="#FFD700" strokeWidth="2.5" strokeLinecap="round" opacity={0.8} />
          );
        })}

        {/* Sun core */}
        <circle cx={SUN_CX} cy={SUN_CY} r="24" fill="#FFD700" filter="url(#sunGlowX)" />
        <circle cx={SUN_CX} cy={SUN_CY} r="16" fill="#FFA500" />

        {/* Energy path guide */}
        <line
          x1={SUN_CX} y1={PARTICLE_Y_START}
          x2={SUN_CX} y2={PARTICLE_Y_END}
          stroke="#FFD700" strokeWidth="1.5"
          strokeDasharray="5 5" opacity={0.2}
        />

        {/* Energy particles */}
        <circle ref={p0Ref} cx={SUN_CX} cy={PARTICLE_Y_START} r="6" fill="#FFD700" opacity={0} />
        <circle ref={p1Ref} cx={SUN_CX} cy={PARTICLE_Y_START} r="6" fill="#FFD700" opacity={0} />
        <circle ref={p2Ref} cx={SUN_CX} cy={PARTICLE_Y_START} r="6" fill="#FFD700" opacity={0} />

        {/* Battery terminal nub */}
        <rect x={BAT_X + 28} y={BAT_Y - 10} width="44" height="10" rx="1"
          fill="#2d3748" stroke="white" strokeWidth="1.5" />

        {/* Battery container */}
        <rect x={BAT_X} y={BAT_Y} width={BAT_W} height={BAT_H}
          rx="2" fill="#111827" stroke="white" strokeWidth="2" />

        {/* Level guides at 25 / 50 / 75 % */}
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

        {/* Battery fill rect */}
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

        {/* SOC percentage text */}
        <text
          ref={battPctRef}
          x={SUN_CX} y={BAT_Y + BAT_H / 2 + 1}
          textAnchor="middle" dominantBaseline="middle"
          fill="white" fontSize="20" fontWeight="bold"
          fontFamily="monospace"
          style={{ mixBlendMode: 'difference' as const }}
        >
          0%
        </text>

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
            {'\u20A9'}
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
