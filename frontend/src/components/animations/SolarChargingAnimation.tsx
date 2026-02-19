/**
 * SolarChargingAnimation - 3-phase loop: charging → converting → discharging
 * Static sun with glow pulse, energy particles, battery fill, and ₩ conversion badge
 */
import { useEffect, useState, useRef } from 'react';
import { Box } from '@chakra-ui/react';
import { motion, useReducedMotion } from 'framer-motion';

type Phase = 'charging' | 'converting' | 'discharging';

const CHARGE_DURATION = 4000;   // ms: SOC 0 → 80
const CONVERT_DURATION = 900;   // ms: flash + ₩ appear
const DISCHARGE_DURATION = 2000; // ms: SOC 80 → 0, ₩ flies up
const PAUSE_DURATION = 800;     // ms: pause before next cycle

const SUN_RAYS = Array.from({ length: 8 }, (_, i) => i);

function getSocFillColor(soc: number): string {
  if (soc >= 80) return '#10B981';
  if (soc >= 50) return '#22C55E';
  if (soc >= 20) return '#F59E0B';
  return '#EF4444';
}

// Battery geometry
const BAT_X = 98;
const BAT_Y = 240;
const BAT_W = 64;
const BAT_H = 116;

export const SolarChargingAnimation = () => {
  const shouldReduceMotion = useReducedMotion();
  const [phase, setPhase] = useState<Phase>('charging');
  const [soc, setSoc] = useState(0);
  const rafRef = useRef<number | null>(null);
  const startRef = useRef<number>(0);
  const socRef = useRef<number>(0);

  useEffect(() => {
    if (shouldReduceMotion) {
      setSoc(50);
      return;
    }

    let currentPhase: Phase = 'charging';
    let phaseStart = performance.now();
    socRef.current = 0;

    function tick(now: number) {
      const elapsed = now - phaseStart;

      if (currentPhase === 'charging') {
        const progress = Math.min(elapsed / CHARGE_DURATION, 1);
        const newSoc = Math.round(progress * 80);
        if (newSoc !== socRef.current) {
          socRef.current = newSoc;
          setSoc(newSoc);
        }
        if (progress >= 1) {
          currentPhase = 'converting';
          setPhase('converting');
          phaseStart = now;
        }
      } else if (currentPhase === 'converting') {
        if (elapsed >= CONVERT_DURATION) {
          currentPhase = 'discharging';
          setPhase('discharging');
          phaseStart = now;
        }
      } else if (currentPhase === 'discharging') {
        const progress = Math.min(elapsed / DISCHARGE_DURATION, 1);
        const newSoc = Math.round(80 * (1 - progress));
        if (newSoc !== socRef.current) {
          socRef.current = newSoc;
          setSoc(newSoc);
        }
        if (progress >= 1) {
          // pause then restart
          setTimeout(() => {
            currentPhase = 'charging';
            socRef.current = 0;
            setSoc(0);
            setPhase('charging');
            phaseStart = performance.now();
            rafRef.current = requestAnimationFrame(tick);
          }, PAUSE_DURATION);
          return; // stop RAF loop during pause
        }
      }

      rafRef.current = requestAnimationFrame(tick);
    }

    rafRef.current = requestAnimationFrame(tick);
    startRef.current = performance.now();

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [shouldReduceMotion]);

  const fillColor = getSocFillColor(soc);
  const fillHeight = (soc / 100) * BAT_H;
  const fillY = BAT_Y + BAT_H - fillHeight;

  const isCharging = phase === 'charging';

  return (
    <Box
      width="260px"
      height="380px"
      bg="spacex.darkGray"
      border="1px solid"
      borderColor="spacex.borderGray"
      display="flex"
      alignItems="center"
      justifyContent="center"
      position="relative"
      overflow="hidden"
    >
      <svg width="260" height="410" viewBox="0 0 260 410">
        <defs>
          <filter id="sunGlow">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* ── Static Sun ── */}
        {/* Glow ring (animated opacity only) */}
        <motion.circle
          cx="130" cy="72" r="36"
          fill="none"
          stroke="#FFD700"
          strokeWidth="2"
          opacity={0}
          animate={shouldReduceMotion ? { opacity: 0 } : { opacity: [0, 0.35, 0] }}
          transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
        />

        {/* Static rays (fixed position, no rotation) */}
        {SUN_RAYS.map((i) => {
          const angle = (i * 45 * Math.PI) / 180;
          const inner = 28;
          const outer = 42;
          const x1 = 130 + inner * Math.cos(angle);
          const y1 = 72 + inner * Math.sin(angle);
          const x2 = 130 + outer * Math.cos(angle);
          const y2 = 72 + outer * Math.sin(angle);
          return (
            <line
              key={i}
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="#FFD700"
              strokeWidth="2.5"
              strokeLinecap="round"
              opacity={0.8}
            />
          );
        })}

        {/* Sun core */}
        <circle cx="130" cy="72" r="20" fill="#FFD700" filter="url(#sunGlow)" />
        <circle cx="130" cy="72" r="14" fill="#FFA500" />

        {/* ── Energy path guide ── */}
        <line
          x1="130" y1="116" x2="130" y2="238"
          stroke="#FFD700" strokeWidth="1.5"
          strokeDasharray="5 5" opacity={0.2}
        />

        {/* ── Energy particles (only during charging) ── */}
        {!shouldReduceMotion && isCharging && [0, 1, 2].map((i) => (
          <motion.circle
            key={i}
            cx="130" r="5"
            fill="#FFD700"
            initial={{ cy: 116, opacity: 0 }}
            animate={{ cy: [116, 238], opacity: [0, 0.9, 0.9, 0] }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'easeInOut',
              delay: i * 0.5,
            }}
          />
        ))}

        {/* ── Battery ── */}
        {/* Terminal nub */}
        <rect x="113" y="232" width="34" height="8" rx="1"
          fill="#2d3748" stroke="white" strokeWidth="1.5" />

        {/* Battery container */}
        <rect x={BAT_X} y={BAT_Y} width={BAT_W} height={BAT_H}
          rx="2" fill="#111827" stroke="white" strokeWidth="2" />

        {/* Level guides */}
        {[25, 50, 75].map((level) => (
          <line
            key={level}
            x1={BAT_X} y1={BAT_Y + BAT_H * (1 - level / 100)}
            x2={BAT_X + BAT_W} y2={BAT_Y + BAT_H * (1 - level / 100)}
            stroke="#4a5568" strokeWidth="1" opacity={0.4}
          />
        ))}

        {/* Battery fill */}
        <clipPath id="battClip">
          <rect x={BAT_X + 2} y={BAT_Y + 2} width={BAT_W - 4} height={BAT_H - 4} />
        </clipPath>
        {fillHeight > 0 && (
          <rect
            x={BAT_X + 2}
            y={fillY + 2}
            width={BAT_W - 4}
            height={Math.max(0, fillHeight - 4)}
            fill={fillColor}
            opacity={0.9}
            clipPath="url(#battClip)"
          />
        )}

        {/* SOC percentage */}
        <text
          x="130" y={BAT_Y + BAT_H / 2 + 1}
          textAnchor="middle" dominantBaseline="middle"
          fill="white" fontSize="17" fontWeight="bold"
          fontFamily="monospace"
          style={{ mixBlendMode: 'difference' }}
        >
          {soc}%
        </text>

        {/* ── ₩ Conversion Badge ── */}
        {!shouldReduceMotion && (phase === 'converting' || phase === 'discharging') && (
          <motion.g
            initial={{ opacity: 0, y: 0 }}
            animate={
              phase === 'converting'
                ? { opacity: 1, y: 0, scale: [0.5, 1.1, 1] }
                : { opacity: [1, 1, 0], y: -70 }
            }
            transition={
              phase === 'converting'
                ? { duration: 0.5, ease: 'backOut' }
                : { duration: DISCHARGE_DURATION / 1000, ease: 'easeOut' }
            }
          >
            {/* Badge background */}
            <circle cx="130" cy="195" r="22" fill="#1c2536" stroke="#FFD700" strokeWidth="2" />
            {/* ₩ symbol */}
            <text
              x="130" y="196"
              textAnchor="middle" dominantBaseline="middle"
              fill="#FFD700" fontSize="22" fontWeight="bold"
              fontFamily="monospace"
            >
              ₩
            </text>
          </motion.g>
        )}

        {/* ── Bottom label ── */}
        <text
          x="130" y="384"
          textAnchor="middle"
          fill="#d1d5db" fontSize="12"
          fontFamily="monospace" letterSpacing="4"
          fontWeight="bold"
        >
          {phase === 'charging' ? 'CHARGING' : phase === 'converting' ? 'CONVERTING' : 'SELLING'}
        </text>
      </svg>
    </Box>
  );
};

export default SolarChargingAnimation;
