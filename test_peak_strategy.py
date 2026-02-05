#!/usr/bin/env python3
"""
Test script for the TEST peak exit strategy using provided sample data.
Standalone implementation - does not require numpy.
"""


class TestPeakExitSimple:
    """Simple implementation of TEST Peak Exit for testing without numpy dependency."""

    def __init__(self, entry_price,
                 ewo_overbought_threshold=0.5,
                 ewo_spread_threshold=0.1,
                 min_profit_pct=0.35,
                 confirmation_bars=2,
                 velocity_lookback=3,
                 overbought_memory=3):  # NEW: how many bars to remember overbought state
        self.entry_price = entry_price
        self.ewo_overbought_threshold = ewo_overbought_threshold
        self.ewo_spread_threshold = ewo_spread_threshold
        self.min_profit_pct = min_profit_pct
        self.confirmation_bars = confirmation_bars
        self.velocity_lookback = velocity_lookback
        self.overbought_memory = overbought_memory

        self.mode = 'watching'
        self.max_option_price = entry_price
        self.max_ewo_fast = 0.0
        self.ewo_history = []
        self.declining_count = 0
        self.peak_ewo_value = 0.0
        self.peak_price = entry_price
        self.bars_since_overbought = 999  # Track bars since last overbought

    def _calculate_profit_pct(self, current_price):
        if self.entry_price <= 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price

    def _calculate_ewo_velocity(self):
        if len(self.ewo_history) < 2:
            return None
        lookback = min(self.velocity_lookback, len(self.ewo_history) - 1)
        recent = self.ewo_history[-lookback:]
        velocities = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        return sum(velocities) / len(velocities) if velocities else None

    def _is_overbought(self, ewo_fast, ewo_slow):
        ewo_spread = ewo_fast - ewo_slow
        return (ewo_fast >= self.ewo_overbought_threshold and
                ewo_spread >= self.ewo_spread_threshold)

    def _was_recently_overbought(self, ewo_fast, ewo_slow):
        """Check if currently or recently overbought."""
        if self._is_overbought(ewo_fast, ewo_slow):
            self.bars_since_overbought = 0
            return True
        self.bars_since_overbought += 1
        return self.bars_since_overbought <= self.overbought_memory

    def update(self, current_option_price, ewo_fast, ewo_slow):
        if current_option_price > self.max_option_price:
            self.max_option_price = current_option_price
        if ewo_fast > self.max_ewo_fast:
            self.max_ewo_fast = ewo_fast

        self.ewo_history.append(ewo_fast)
        if len(self.ewo_history) > 20:
            self.ewo_history.pop(0)

        profit_pct = self._calculate_profit_pct(current_option_price)
        ewo_spread = ewo_fast - ewo_slow
        ewo_velocity = self._calculate_ewo_velocity()

        triggered = False
        sell_reason = None

        if len(self.ewo_history) >= 2:
            if self.ewo_history[-1] < self.ewo_history[-2]:
                self.declining_count += 1
            else:
                self.declining_count = 0

        # Check recently overbought (updates internal counter)
        recently_overbought = self._was_recently_overbought(ewo_fast, ewo_slow)

        if self.mode == 'watching':
            if (profit_pct >= self.min_profit_pct and
                    recently_overbought and
                    ewo_velocity is not None and ewo_velocity < 0):
                self.mode = 'peak_detected'
                self.peak_ewo_value = self.max_ewo_fast
                self.peak_price = self.max_option_price

        elif self.mode == 'peak_detected':
            if self.declining_count >= self.confirmation_bars:
                self.mode = 'confirming'
            if ewo_fast > self.peak_ewo_value * 1.02:
                self.mode = 'watching'
                self.declining_count = 0

        elif self.mode == 'confirming':
            if current_option_price < self.peak_price * 0.995:
                self.mode = 'triggered'
                triggered = True
                sell_reason = 'test_peak_momentum_reversal'
            if ewo_fast > self.peak_ewo_value * 1.02:
                self.mode = 'watching'
                self.declining_count = 0

        elif self.mode == 'triggered':
            triggered = True
            sell_reason = 'test_peak_momentum_reversal'

        return {
            'triggered': triggered,
            'sell_reason': sell_reason,
            'mode': self.mode,
            'max_price': self.max_option_price,
            'ewo_velocity': ewo_velocity,
            'ewo_spread': ewo_spread,
            'declining_count': self.declining_count,
            'peak_ewo_value': self.peak_ewo_value,
            'peak_price': self.peak_price,
            'profit_pct': profit_pct,
            'is_overbought': self._is_overbought(ewo_fast, ewo_slow),
            'recently_overbought': recently_overbought
        }


# Sample data from user
data = """
2026-02-04 10:40:00-05:00	TRUE	$322.73	$323.25	$322.60	$1.50	+0.0%	$1.50	$1.05	initial		initial	FALSE	$321.88	$322.75	0.437	0.747	$0.45
2026-02-04 10:41:00-05:00	TRUE	$322.57	$322.73	$322.50	$1.42	-5.3%	$1.50	$1.05	initial		initial	FALSE	$321.89	$322.74	0.268	0.734	$0.37
2026-02-04 10:42:00-05:00	TRUE	$322.38	$322.55	$322.38	$1.33	-11.6%	$1.50	$1.05	initial		initial	FALSE	$321.89	$322.72	0.105	0.706	$0.28
2026-02-04 10:43:00-05:00	TRUE	$322.61	$322.83	$322.38	$1.44	-4.0%	$1.50	$1.05	initial		initial	FALSE	$321.91	$322.71	0.063	0.667	$0.39
2026-02-04 10:44:00-05:00	TRUE	$322.56	$322.67	$322.43	$1.42	-5.6%	$1.50	$1.05	initial		initial	FALSE	$321.91	$322.70	0.022	0.622	$0.37
2026-02-04 10:45:00-05:00	TRUE	$322.66	$322.86	$322.52	$1.47	-2.3%	$1.50	$1.05	initial		initial	FALSE	$321.93	$322.70	0.024	0.570	$0.42
2026-02-04 10:46:00-05:00	TRUE	$322.69	$322.85	$322.62	$1.48	-1.3%	$1.50	$1.05	initial		initial	FALSE	$321.93	$322.70	0.033	0.509	$0.43
2026-02-04 10:47:00-05:00	TRUE	$322.58	$322.71	$322.47	$1.43	-5.0%	$1.50	$1.05	initial		initial	FALSE	$321.94	$322.69	0.007	0.447	$0.38
2026-02-04 10:48:00-05:00	TRUE	$322.74	$322.80	$322.58	$1.51	+0.3%	$1.51	$1.05	initial		initial	TRUE	$321.94	$322.69	0.035	0.385	$0.46
2026-02-04 10:49:00-05:00	TRUE	$322.83	$322.89	$322.77	$1.55	+3.5%	$1.55	$1.05	initial		initial	TRUE	$321.95	$322.70	0.079	0.328	$0.50
2026-02-04 10:50:00-05:00	TRUE	$322.70	$323.00	$322.70	$1.49	-1.0%	$1.55	$1.05	initial		initial	FALSE	$321.95	$322.70	0.067	0.278	$0.44
2026-02-04 10:51:00-05:00	TRUE	$322.70	$322.80	$322.55	$1.49	-1.0%	$1.55	$1.05	initial		initial	FALSE	$321.96	$322.70	0.059	0.227	$0.44
2026-02-04 10:52:00-05:00	TRUE	$322.91	$322.91	$322.76	$1.59	+6.0%	$1.59	$1.05	initial		initial	TRUE	$321.96	$322.71	0.111	0.178	$0.54
2026-02-04 10:53:00-05:00	TRUE	$323.11	$323.13	$322.92	$1.69	+12.6%	$1.69	$1.05	initial		initial	TRUE	$321.97	$322.74	0.197	0.142	$0.64
2026-02-04 10:54:00-05:00	TRUE	$323.03	$323.12	$322.98	$1.65	+9.9%	$1.69	$1.05	initial		initial	TRUE	$321.97	$322.76	0.225	0.116	$0.60
2026-02-04 10:55:00-05:00	TRUE	$323.46	$323.50	$323.08	$1.87	+24.2%	$1.87	$1.05	initial		initial	TRUE	$321.98	$322.80	0.358	0.110	$0.81
2026-02-04 10:56:00-05:00	TRUE	$323.61	$323.61	$323.33	$1.94	+29.2%	$1.94	$1.05	initial		initial	TRUE	$321.99	$322.86	0.477	0.124	$0.89
2026-02-04 10:57:00-05:00	TRUE	$323.79	$323.88	$323.48	$2.03	+35.2%	$2.03	$1.05	initial	$1.65	tier_35	TRUE	$322.00	$322.92	0.593	0.157	$0.98
2026-02-04 10:58:00-05:00	TRUE	$323.47	$323.86	$323.33	$1.87	+24.7%	$2.03	$1.05	initial	$1.65	tier_35	TRUE	$322.01	$322.95	0.567	0.190	$0.82
2026-02-04 10:59:00-05:00	TRUE	$323.87	$323.88	$323.44	$2.07	+37.9%	$2.07	$1.05	initial	$1.65	tier_35	TRUE	$322.04	$323.01	0.651	0.232	$1.02
2026-02-04 11:00:00-05:00	TRUE	$324.22	$324.22	$323.84	$2.25	+49.5%	$2.25	$1.05	initial	$1.65	tier_35	TRUE	$322.06	$323.09	0.788	0.283	$1.19
2026-02-04 11:01:00-05:00	TRUE	$324.30	$324.31	$323.93	$2.29	+52.2%	$2.29	$1.60	trailing	$1.80	tier_50	TRUE	$322.11	$323.17	0.883	0.340	$0.69
2026-02-04 11:02:00-05:00	TRUE	$324.24	$324.40	$324.18	$2.25	+50.1%	$2.29	$1.60	trailing	$1.80	tier_50	TRUE	$322.13	$323.24	0.908	0.400	$0.65
2026-02-04 11:03:00-05:00	TRUE	$324.17	$324.24	$324.04	$2.22	+47.9%	$2.29	$1.60	trailing	$1.80	tier_50	TRUE	$322.15	$323.30	0.889	0.457	$0.62
2026-02-04 11:04:00-05:00	TRUE	$324.25	$324.28	$324.06	$2.26	+50.5%	$2.29	$1.60	trailing	$1.80	tier_50	TRUE	$322.17	$323.36	0.882	0.510	$0.66
2026-02-04 11:05:00-05:00	TRUE	$324.39	$324.42	$324.23	$2.33	+55.0%	$2.33	$1.63	trailing	$1.80	tier_50	TRUE	$322.19	$323.42	0.900	0.566	$0.70
2026-02-04 11:06:00-05:00	TRUE	$324.14	$324.42	$324.14	$2.21	+47.0%	$2.33	$1.63	trailing	$1.80	tier_50	TRUE	$322.20	$323.47	0.827	0.617	$0.58
2026-02-04 11:07:00-05:00	TRUE	$323.84	$324.22	$323.84	$2.05	+36.9%	$2.33	$1.63	trailing	$1.80	tier_50	TRUE	$322.24	$323.49	0.681	0.655	$0.43
2026-02-04 11:08:00-05:00	TRUE	$323.89	$323.94	$323.70	$2.08	+38.7%	$2.33	$1.63	trailing	$1.80	tier_50	TRUE	$322.26	$323.52	0.592	0.682	$0.45
2026-02-04 11:09:00-05:00	TRUE	$323.93	$324.03	$323.86	$2.10	+40.0%	$2.33	$1.63	trailing	$1.80	tier_50	TRUE	$322.27	$323.55	0.536	0.702	$0.47
2026-02-04 11:10:00-05:00	TRUE	$324.07	$324.07	$323.84	$2.17	+44.5%	$2.33	$1.63	trailing	$1.80	tier_50	TRUE	$322.29	$323.58	0.529	0.714	$0.54
2026-02-04 11:11:00-05:00	TRUE	$324.02	$324.13	$323.90	$2.14	+42.9%	$2.33	$1.63	trailing	$1.80	tier_50	TRUE	$322.30	$323.61	0.500	0.715	$0.52
2026-02-04 11:12:00-05:00	TRUE	$324.23	$324.24	$324.02	$2.25	+49.9%	$2.33	$1.63	trailing	$1.80	tier_50	TRUE	$322.33	$323.65	0.531	0.711	$0.62
2026-02-04 11:13:00-05:00	TRUE	$324.19	$324.26	$324.10	$2.23	+48.5%	$2.33	$1.63	trailing	$1.80	tier_50	TRUE	$322.34	$323.68	0.530	0.708	$0.60
2026-02-04 11:14:00-05:00	TRUE	$324.43	$324.50	$324.19	$2.35	+56.7%	$2.35	$1.65	trailing	$1.80	tier_50	TRUE	$322.35	$323.73	0.587	0.704	$0.71
2026-02-04 11:15:00-05:00	TRUE	$324.41	$324.55	$324.34	$2.34	+55.9%	$2.35	$1.65	trailing	$1.80	tier_50	TRUE	$322.39	$323.78	0.606	0.692	$0.69
2026-02-04 11:16:00-05:00	TRUE	$324.54	$324.60	$324.38	$2.41	+60.2%	$2.41	$1.68	trailing	$1.80	tier_50	TRUE	$322.40	$323.83	0.642	0.676	$0.72
2026-02-04 11:17:00-05:00	TRUE	$324.71	$324.71	$324.49	$2.49	+65.9%	$2.49	$1.74	trailing	$1.80	tier_50	TRUE	$322.42	$323.88	0.700	0.662	$0.75
2026-02-04 11:18:00-05:00	TRUE	$324.92	$324.92	$324.68	$2.60	+72.9%	$2.60	$1.82	trailing	$1.80	tier_50	TRUE	$322.44	$323.95	0.782	0.655	$0.78
2026-02-04 11:19:00-05:00	TRUE	$324.76	$324.92	$324.76	$2.52	+67.6%	$2.60	$1.82	trailing	$1.80	tier_50	TRUE	$322.46	$324.00	0.775	0.648	$0.70
2026-02-04 11:20:00-05:00	TRUE	$324.48	$324.76	$324.42	$2.37	+58.0%	$2.60	$1.82	trailing	$1.80	tier_50	TRUE	$322.47	$324.03	0.677	0.633	$0.56
2026-02-04 11:21:00-05:00	TRUE	$324.54	$324.59	$324.48	$2.40	+60.0%	$2.60	$1.82	trailing	$1.80	tier_50	TRUE	$322.48	$324.07	0.619	0.619	$0.59
2026-02-04 11:22:00-05:00	TRUE	$324.66	$324.75	$324.57	$2.47	+64.2%	$2.60	$1.82	trailing	$1.80	tier_50	TRUE	$322.48	$324.10	0.606	0.614	$0.65
2026-02-04 11:23:00-05:00	TRUE	$324.69	$324.75	$324.59	$2.48	+65.2%	$2.60	$1.82	trailing	$1.80	tier_50	TRUE	$322.51	$324.14	0.595	0.614	$0.66
2026-02-04 11:24:00-05:00	TRUE	$324.80	$324.86	$324.49	$2.54	+69.1%	$2.60	$1.82	trailing	$1.80	tier_50	TRUE	$322.53	$324.18	0.609	0.619	$0.72
2026-02-04 11:25:00-05:00	TRUE	$324.71	$324.86	$324.71	$2.49	+66.1%	$2.60	$1.82	trailing	$1.80	tier_50	TRUE	$322.54	$324.22	0.582	0.623	$0.68
2026-02-04 11:26:00-05:00	TRUE	$324.85	$324.85	$324.70	$2.56	+70.6%	$2.60	$1.82	trailing	$1.80	tier_50	TRUE	$322.56	$324.26	0.591	0.629	$0.74
2026-02-04 11:27:00-05:00	TRUE	$325.15	$325.17	$324.89	$2.71	+80.8%	$2.71	$1.90	trailing	$2.03	tier_75	TRUE	$322.67	$324.32	0.671	0.638	$0.81
2026-02-04 11:28:00-05:00	TRUE	$325.14	$325.32	$325.12	$2.70	+80.1%	$2.71	$1.90	trailing	$2.03	tier_75	TRUE	$322.68	$324.37	0.703	0.650	$0.80
2026-02-04 11:29:00-05:00	TRUE	$325.15	$325.32	$325.14	$2.71	+80.6%	$2.71	$1.90	trailing	$2.03	tier_75	TRUE	$322.69	$324.42	0.714	0.658	$0.81
2026-02-04 11:30:00-05:00	TRUE	$325.18	$325.22	$325.08	$2.73	+81.6%	$2.73	$1.91	trailing	$2.03	tier_75	TRUE	$322.71	$324.47	0.717	0.666	$0.82
2026-02-04 11:31:00-05:00	TRUE	$325.30	$325.30	$325.15	$2.79	+85.6%	$2.79	$1.95	trailing	$2.03	tier_75	TRUE	$322.72	$324.52	0.738	0.672	$0.84
2026-02-04 11:32:00-05:00	TRUE	$325.43	$325.45	$325.27	$2.85	+90.0%	$2.85	$2.00	trailing	$2.03	tier_75	TRUE	$322.73	$324.58	0.774	0.677	$0.86
2026-02-04 11:33:00-05:00	TRUE	$325.55	$325.59	$325.39	$2.92	+94.2%	$2.92	$2.04	trailing	$2.03	tier_75	TRUE	$322.75	$324.64	0.817	0.679	$0.87
2026-02-04 11:34:00-05:00	TRUE	$325.43	$325.61	$325.40	$2.85	+90.0%	$2.92	$2.04	trailing	$2.03	tier_75	TRUE	$322.76	$324.69	0.795	0.680	$0.81
2026-02-04 11:35:00-05:00	TRUE	$325.50	$325.55	$325.39	$2.89	+92.4%	$2.92	$2.04	trailing	$2.03	tier_75	TRUE	$322.77	$324.75	0.785	0.688	$0.85
2026-02-04 11:36:00-05:00	TRUE	$325.49	$325.57	$325.41	$2.88	+92.0%	$2.92	$2.04	trailing	$2.03	tier_75	TRUE	$322.79	$324.79	0.761	0.697	$0.84
2026-02-04 11:37:00-05:00	TRUE	$325.53	$325.57	$325.41	$2.90	+93.4%	$2.92	$2.04	trailing	$2.03	tier_75	TRUE	$322.80	$324.84	0.744	0.706	$0.86
2026-02-04 11:38:00-05:00	TRUE	$325.56	$325.67	$325.44	$2.92	+94.4%	$2.92	$2.04	trailing	$2.03	tier_75	TRUE	$322.81	$324.89	0.727	0.715	$0.88
2026-02-04 11:39:00-05:00	TRUE	$325.73	$325.73	$325.52	$3.01	+100.2%	$3.01	$2.10	trailing	$2.25	tier_100	TRUE	$322.83	$324.94	0.752	0.725	$0.90
2026-02-04 11:40:00-05:00	TRUE	$325.61	$325.76	$325.52	$2.95	+96.2%	$3.01	$2.10	trailing	$2.25	tier_100	TRUE	$322.85	$324.99	0.720	0.734	$0.84
2026-02-04 11:41:00-05:00	TRUE	$326.08	$326.14	$325.61	$3.18	+111.9%	$3.18	$2.23	trailing	$2.25	tier_100	TRUE	$322.90	$325.06	0.816	0.749	$0.95
2026-02-04 11:42:00-05:00	TRUE	$326.01	$326.17	$325.93	$3.15	+109.5%	$3.18	$2.23	trailing	$2.25	tier_100	TRUE	$322.96	$325.12	0.842	0.760	$0.92
2026-02-04 11:43:00-05:00	TRUE	$325.99	$326.08	$325.96	$3.14	+108.8%	$3.18	$2.23	trailing	$2.25	tier_100	TRUE	$322.97	$325.17	0.837	0.769	$0.91
2026-02-04 11:44:00-05:00	TRUE	$326.06	$326.09	$325.93	$3.17	+111.2%	$3.18	$2.23	trailing	$2.25	tier_100	TRUE	$322.99	$325.23	0.838	0.777	$0.94
2026-02-04 11:45:00-05:00	TRUE	$326.39	$326.41	$326.03	$3.34	+122.3%	$3.34	$2.34	trailing	$2.34	tier_100	TRUE	$323.01	$325.31	0.914	0.791	$1.00
2026-02-04 11:46:00-05:00	TRUE	$326.36	$326.51	$326.27	$3.32	+121.3%	$3.34	$2.34	trailing	$2.34	tier_100	TRUE	$323.05	$325.37	0.938	0.804	$0.99
2026-02-04 11:47:00-05:00	TRUE	$326.50	$326.58	$326.37	$3.39	+126.0%	$3.39	$2.38	trailing	$2.38	tier_125	TRUE	$323.07	$325.45	0.974	0.817	$1.02
2026-02-04 11:48:00-05:00	TRUE	$326.36	$326.54	$326.34	$3.32	+121.3%	$3.39	$2.38	trailing	$2.38	tier_125	TRUE	$323.09	$325.51	0.939	0.825	$0.95
2026-02-04 11:49:00-05:00	TRUE	$326.35	$326.51	$326.24	$3.32	+120.9%	$3.39	$2.38	trailing	$2.38	tier_125	TRUE	$323.12	$325.56	0.898	0.832	$0.94
2026-02-04 11:50:00-05:00	TRUE	$326.26	$326.39	$326.21	$3.27	+117.9%	$3.39	$2.38	trailing	$2.38	tier_125	TRUE	$323.13	$325.61	0.830	0.835	$0.90
2026-02-04 11:51:00-05:00	TRUE	$326.00	$326.46	$325.95	$3.14	+109.2%	$3.39	$2.38	trailing	$2.38	tier_125	TRUE	$323.17	$325.63	0.699	0.831	$0.77
2026-02-04 11:52:00-05:00	TRUE	$326.06	$326.18	$325.93	$3.17	+111.2%	$3.39	$2.38	trailing	$2.38	tier_125	TRUE	$323.18	$325.66	0.621	0.823	$0.80
2026-02-04 11:53:00-05:00	TRUE	$326.14	$326.22	$326.03	$3.21	+113.9%	$3.39	$2.38	trailing	$2.38	tier_125	TRUE	$323.19	$325.69	0.582	0.813	$0.84
2026-02-04 11:54:00-05:00	TRUE	$326.00	$326.08	$325.96	$3.14	+109.3%	$3.39	$2.38	trailing	$2.38	tier_125	TRUE	$323.21	$325.71	0.510	0.797	$0.77
2026-02-04 11:55:00-05:00	TRUE	$325.96	$326.08	$325.92	$3.12	+108.0%	$3.39	$2.38	trailing	$2.38	tier_125	TRUE	$323.23	$325.73	0.443	0.779	$0.75
2026-02-04 11:56:00-05:00	TRUE	$326.05	$326.08	$325.93	$3.17	+110.8%	$3.39	$2.38	trailing	$2.38	tier_125	TRUE	$323.24	$325.75	0.417	0.752	$0.79
2026-02-04 11:57:00-05:00	TRUE	$325.88	$326.05	$325.77	$3.08	+105.1%	$3.39	$2.38	trailing	$2.38	tier_125	TRUE	$323.26	$325.76	0.346	0.719	$0.70
2026-02-04 11:58:00-05:00	TRUE	$325.57	$326.04	$325.35	$2.92	+94.7%	$3.39	$2.38	trailing	$2.38	tier_125	FALSE	$323.27	$325.74	0.208	0.677	$0.55
"""


def parse_data(data_str):
    """Parse the tab-separated data into structured records."""
    records = []
    for line in data_str.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 17:
            continue

        timestamp = parts[0].strip()
        in_position = parts[1].strip() == 'TRUE'

        # Parse option price (remove $ and convert)
        option_price_str = parts[5].strip().replace('$', '')
        option_price = float(option_price_str)

        # Parse EWO fast and slow (columns 16 and 17, 0-indexed 15 and 16)
        ewo_fast = float(parts[15].strip())
        ewo_slow = float(parts[16].strip())

        records.append({
            'timestamp': timestamp,
            'in_position': in_position,
            'option_price': option_price,
            'ewo_fast': ewo_fast,
            'ewo_slow': ewo_slow
        })

    return records


def test_run(strategy, records, entry_price, verbose=False):
    """Run a single test with given strategy."""
    exit_triggered = False
    exit_time = None
    exit_price = None

    for record in records:
        if not record['in_position']:
            continue

        result = strategy.update(
            current_option_price=record['option_price'],
            ewo_fast=record['ewo_fast'],
            ewo_slow=record['ewo_slow']
        )

        if verbose and result['mode'] != 'watching':
            profit_pct = (record['option_price'] - entry_price) / entry_price * 100
            print(f"  {record['timestamp'][-14:-6]} ${record['option_price']:.2f} +{profit_pct:.0f}% EWO:{record['ewo_fast']:.3f} Mode:{result['mode']}")

        if result['triggered'] and not exit_triggered:
            exit_triggered = True
            exit_time = record['timestamp']
            exit_price = record['option_price']

    return exit_triggered, exit_time, exit_price


def run_test():
    """Run the TEST peak exit strategy on sample data."""
    records = parse_data(data)
    entry_price = 1.50

    print("=" * 80)
    print("TEST Peak Exit Strategy - Parameter Comparison")
    print("=" * 80)
    print(f"Entry Price:   ${entry_price:.2f} at 10:40")
    print(f"Peak Price:    $3.39 (+126.0%) at 11:47")
    print(f"Trailing Exit: $2.92 (+94.7%) at 11:58")
    print("=" * 80)

    # Test with different parameter sets
    param_sets = [
        {'name': 'Default (ewo>=0.5, profit>=35%)', 'ewo_ob': 0.5, 'spread': 0.1, 'profit': 0.35, 'confirm': 2, 'memory': 0},
        {'name': 'EWO>=0.9, profit>=100%, memory=3', 'ewo_ob': 0.9, 'spread': 0.1, 'profit': 1.00, 'confirm': 2, 'memory': 3},
        {'name': 'EWO>=0.9, profit>=100%, memory=5', 'ewo_ob': 0.9, 'spread': 0.1, 'profit': 1.00, 'confirm': 2, 'memory': 5},
        {'name': 'EWO>=0.85, profit>=100%, memory=3', 'ewo_ob': 0.85, 'spread': 0.1, 'profit': 1.00, 'confirm': 2, 'memory': 3},
        {'name': 'EWO>=0.8, profit>=75%, memory=3', 'ewo_ob': 0.8, 'spread': 0.1, 'profit': 0.75, 'confirm': 2, 'memory': 3},
        {'name': 'EWO>=0.8, profit>=100%, memory=5', 'ewo_ob': 0.8, 'spread': 0.1, 'profit': 1.00, 'confirm': 2, 'memory': 5},
    ]

    print(f"\n{'Parameter Set':<50} {'Exit Time':>12} {'Exit Price':>12} {'Profit':>10} {'vs Trail':>10}")
    print("-" * 100)

    for params in param_sets:
        strategy = TestPeakExitSimple(
            entry_price=entry_price,
            ewo_overbought_threshold=params['ewo_ob'],
            ewo_spread_threshold=params['spread'],
            min_profit_pct=params['profit'],
            confirmation_bars=params['confirm'],
            velocity_lookback=3,
            overbought_memory=params['memory']
        )

        exit_triggered, exit_time, exit_price = test_run(strategy, records, entry_price)

        if exit_triggered:
            exit_profit = (exit_price - entry_price) / entry_price * 100
            vs_trailing = exit_profit - 94.7
            vs_str = f"+{vs_trailing:.1f}%" if vs_trailing > 0 else f"{vs_trailing:.1f}%"
            print(f"{params['name']:<50} {exit_time[-14:-6]:>12} ${exit_price:>10.2f} {exit_profit:>9.1f}% {vs_str:>10}")
        else:
            print(f"{params['name']:<50} {'No trigger':>12} {'-':>12} {'-':>10} {'-':>10}")

    # Detailed trace for debugging
    print("\n" + "=" * 80)
    print("Debug Trace: ewo>=0.9, profit>=100%, memory=3")
    print("Showing key bars around peak (11:45-11:55)")
    print("=" * 80)

    strategy = TestPeakExitSimple(
        entry_price=entry_price,
        ewo_overbought_threshold=0.9,
        ewo_spread_threshold=0.1,
        min_profit_pct=1.00,
        confirmation_bars=2,
        velocity_lookback=3,
        overbought_memory=3
    )

    print(f"{'Time':<10} {'Price':>6} {'Profit':>7} {'EWO_F':>6} {'Spread':>6} {'Vel':>7} {'OB':>4} {'R_OB':>5} {'Mode':<15}")
    print("-" * 85)

    for record in records:
        if not record['in_position']:
            continue

        result = strategy.update(
            current_option_price=record['option_price'],
            ewo_fast=record['ewo_fast'],
            ewo_slow=record['ewo_slow']
        )

        time_str = record['timestamp'][-14:-6]
        profit_pct = (record['option_price'] - entry_price) / entry_price * 100
        spread = record['ewo_fast'] - record['ewo_slow']
        vel = result['ewo_velocity']
        vel_str = f"{vel:.3f}" if vel is not None else "N/A"
        ob = "Yes" if result['is_overbought'] else "No"
        r_ob = "Yes" if result['recently_overbought'] else "No"

        # Show rows around the peak (11:39 to 11:55)
        if "11:39" <= time_str <= "11:55" or result['mode'] != 'watching':
            print(f"{time_str:<10} ${record['option_price']:>5.2f} {profit_pct:>6.1f}% {record['ewo_fast']:>6.3f} {spread:>6.3f} {vel_str:>7} {ob:>4} {r_ob:>5} {result['mode']:<15}")


if __name__ == '__main__':
    run_test()
