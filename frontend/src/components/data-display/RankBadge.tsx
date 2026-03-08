/**
 * RankBadge - 1st/2nd/3rd rank display
 */
import { Badge } from '@chakra-ui/react';

interface RankBadgeProps {
  rank: number;
  isBest?: boolean;
}

const RANK_STYLES: Record<number, { bg: string; color: string; label: string }> = {
  1: { bg: 'solar.gold', color: 'black', label: 'BEST' },
  2: { bg: 'whiteAlpha.200', color: 'spacex.textGray', label: '2ND' },
  3: { bg: 'whiteAlpha.100', color: 'spacex.textGray', label: '3RD' },
};

export const RankBadge = ({ rank, isBest }: RankBadgeProps) => {
  const style = RANK_STYLES[rank] || RANK_STYLES[3];

  return (
    <Badge
      bg={isBest ? 'solar.gold' : style.bg}
      color={isBest ? 'black' : style.color}
      fontSize="xs"
      px={3}
      py={1}
      borderRadius="0"
      fontWeight="bold"
      letterSpacing="wider"
    >
      {isBest ? 'BEST' : style.label}
    </Badge>
  );
};
