# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

import numpy as np

from artiq.language.core import *
from artiq.language.types import *
from artiq.language.units import *

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager


class AD9912(DaxSimDevice):

    def __init__(self, dmgr, chip_select, cpld_device, sw_device=None, pll_n=10, **kwargs):
        # Call super
        super(AD9912, self).__init__(dmgr, **kwargs)

        # Register signals
        self._signal_manager = get_signal_manager()
        self._init = self._signal_manager.register(self, 'init', bool, size=1)
        self._freq = self._signal_manager.register(self, 'freq', float)
        self._phase = self._signal_manager.register(self, 'phase', float)

        # CPLD device
        self.cpld = dmgr.get(cpld_device)
        # Chip select
        assert 4 <= chip_select <= 7
        self.chip_select = chip_select
        # Switch device
        if sw_device:
            self.sw = dmgr.get(sw_device)

        # Store attributes (from ARTIQ code)
        sysclk = self.cpld.refclk / [1, 1, 2, 4][self.cpld.clk_div] * pll_n
        assert sysclk <= 1e9
        self.ftw_per_hz = 1 / sysclk * (np.int64(1) << 48)

    @kernel
    def write(self, addr, data, length):
        raise NotImplementedError

    @kernel
    def read(self, addr, length):
        raise NotImplementedError

    @kernel
    def init(self):
        # Delays from ARTIQ code
        delay(50 * us)
        delay(1 * ms)
        self._signal_manager.event(self._init, 1)

    @kernel
    def set_att_mu(self, att):
        self.cpld.set_att_mu(self.chip_select - 4, att)

    @kernel
    def set_att(self, att):
        self.cpld.set_att(self.chip_select - 4, att)

    # noinspection PyShadowingBuiltins
    @kernel
    def set_mu(self, ftw, pow):
        phase = pow / (1 << 14)  # Inverted turns_to_pow()
        self.set(self.ftw_to_frequency(ftw), phase)

    @portable(flags={"fast-math"})
    def frequency_to_ftw(self, frequency) -> TInt64:
        return np.int64(round(float(self.ftw_per_hz * frequency))) & ((np.int64(1) << 48) - 1)

    @portable(flags={"fast-math"})
    def ftw_to_frequency(self, ftw):
        return ftw / self.ftw_per_hz

    @portable(flags={"fast-math"})
    def turns_to_pow(self, phase) -> TInt32:
        return np.int32(round(float((1 << 14) * phase))) & np.int32(0xffff)

    @kernel
    def set(self, frequency, phase=0.0):
        assert 0 * MHz <= frequency <= 400 * MHz, 'Frequency out of range'
        assert 0.0 <= phase < 1.0, 'Phase out of range'
        self._signal_manager.event(self._freq, float(frequency))
        self._signal_manager.event(self._phase, float(phase))

    @kernel
    def cfg_sw(self, state):
        self.cpld.cfg_sw(self.chip_select - 4, state)
