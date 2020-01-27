import numpy as np

from dax.base import *
import dax.util.units


class RtioStressModule(DaxModule):
    """Module to stress the RTIO output system."""

    # System keys
    THROUGHPUT_PERIOD_KEY = 'min_period'
    THROUGHPUT_BURST_KEY = 'max_burst'

    # Unique DMA tags
    DMA_BURST = 'rtio_stress_burst'

    def build(self, ttl_out):
        # TTL output device
        self.setattr_device(ttl_out, 'ttl_out')

    def load_module(self):
        # Load throughput parameters
        self.setattr_dataset_sys(self.THROUGHPUT_PERIOD_KEY)
        self.setattr_dataset_sys(self.THROUGHPUT_BURST_KEY)

        # Cap burst size to prevent too long DMA recordings, resulting in a connection timeout
        self.max_burst = min(self.max_burst, 40000)

    @kernel
    def init_module(self):
        # Break realtime to get some slack
        self.core.break_realtime()

        # Set TTL direction
        self.ttl_out.output()

        with self.core_dma.record(self.DMA_BURST):
            # Record the DMA burst trace
            for _ in range(self.max_burst // 2):
                delay(self.min_period / 2)
                self.ttl_out.on()
                delay(self.min_period / 2)
                self.ttl_out.off()

    def post_init_module(self):
        # Obtain DMA handle
        self.burst_dma_handle = self.core_dma.get_handle(self.DMA_BURST)
        self.update_kernel_invariants('burst_dma_handle')

    """Module functionality"""

    @kernel
    def on(self):
        self.ttl_out.on()

    @kernel
    def off(self):
        self.ttl_out.off()

    @kernel
    def pulse(self, duration):
        self.ttl_out.pulse(duration)

    @kernel
    def pulse_mu(self, duration):
        self.ttl_out.pulse_mu(duration)

    @kernel
    def burst(self):
        for _ in range(self.max_burst * 16):
            delay(self.min_period * 2)
            self.ttl_out.on()
            delay(self.min_period * 2)
            self.ttl_out.off()

    @kernel
    def burst_dma(self):
        for _ in range(128):
            self.core_dma.playback_handle(self.burst_dma_handle)

    """Calibrate throughput"""

    def calibrate_throughput(self, period_scan, num_samples, num_events, no_underflow_cutoff):
        # Check arguments
        if not num_samples > 0:
            msg = 'Number of samples must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not num_events > 0:
            msg = 'Number of events must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not no_underflow_cutoff > 0:
            msg = 'No underflow cutoff must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)

        # Sort scan (in-place)
        period_scan.sort()

        # Store input values in dataset
        self.set_dataset('period_scan', period_scan)
        self.set_dataset('num_samples', num_samples)
        self.set_dataset('num_events', num_events)
        self.set_dataset('no_underflow_cutoff', no_underflow_cutoff)

        # Run kernel
        self._calibrate_throughput(period_scan, num_samples, num_events, no_underflow_cutoff)

        # Get results
        no_underflow_count = self.get_dataset('no_underflow_count')
        underflow_flag = self.get_dataset('underflow_flag')
        last_period = self.get_dataset('last_period')

        # Process results directly (next experiment might need these values)
        if no_underflow_count == 0:
            # Last data point was an underflow, assuming all data points raised an underflow
            self.logger.warning('Could not determine throughput: All data points raised an underflow exception')
        elif not underflow_flag:
            # No underflow occurred
            self.logger.warning('Could not determine throughput: No data points raised an underflow exception')
        else:
            # Store result in system dataset
            self.set_dataset_sys(self.THROUGHPUT_PERIOD_KEY, last_period)

    @kernel
    def _calibrate_throughput(self, period_scan, num_samples, num_events, no_underflow_cutoff):
        # Storage for last period
        last_period = 0.0
        # Count of last period without underflow
        no_underflow_count = 0
        # A flag to mark if at least one underflow happened
        underflow_flag = False

        # Iterate over scan
        for current_period in period_scan:
            try:
                # Convert time and start spawning events
                self._spawn_events(current_period, num_samples, num_events)
            except RTIOUnderflow:
                # Set underflow flag
                underflow_flag = True
                # Reset counter
                no_underflow_count = 0
            else:
                if no_underflow_count == 0:
                    # Store the period that works
                    last_period = current_period

                # Increment counter
                no_underflow_count += 1

                if no_underflow_count >= no_underflow_cutoff:
                    # Cutoff reached, stop testing
                    break

        # Store results
        self.set_dataset('no_underflow_count', no_underflow_count)
        self.set_dataset('underflow_flag', underflow_flag)
        self.set_dataset('last_period', last_period)

    @kernel
    def _spawn_events(self, period, num_samples, num_events):
        # Convert period to machine units
        period_mu = self.core.seconds_to_mu(period)
        # Scale number of events
        num_events //= 2

        # Iterate over number of samples
        for _ in range(num_samples):
            # RTIO reset
            self.core.reset()
            self.ttl_out.off()

            # Spawn events, could throw RTIOUnderflow
            for _ in range(num_events):
                delay_mu(period_mu)
                self.ttl_out.on()
                delay_mu(period_mu)
                self.ttl_out.off()

            # RTIO sync
            self.core.wait_until_mu(now_mu())

    """Calibrate throughput burst"""

    def calibrate_throughput_burst(self, num_events_scan, num_samples, period_step, no_underflow_cutoff,
                                   num_step_cutoff):
        # Get current period
        current_period = self.get_dataset_sys(self.THROUGHPUT_PERIOD_KEY)

        # Check arguments
        if not num_samples > 0:
            msg = 'Number of samples must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not period_step > 0.0:
            msg = 'Period step for throughput burst calibration must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not no_underflow_cutoff > 0:
            msg = 'No underflow cutoff must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)

        # Sort scan descending (in-place)
        num_events_scan.sort(reverse=True)

        # Store input values in dataset
        self.set_dataset('num_events_scan', num_events_scan)
        self.set_dataset('num_samples', num_samples)
        self.set_dataset('period_step', period_step)
        self.set_dataset('no_underflow_cutoff', no_underflow_cutoff)
        self.set_dataset('num_step_cutoff', num_step_cutoff)

        # Message starting period
        self._message_current_period(current_period)

        # Run kernel
        self._calibrate_throughput_burst(num_events_scan, num_samples, current_period, period_step,
                                         no_underflow_cutoff, num_step_cutoff)

        # Get results
        no_underflow_count = self.get_dataset('no_underflow_count')
        underflow_flag = self.get_dataset('underflow_flag')
        last_num_events = self.get_dataset('last_num_events')

        # Process results directly (next experiment might need these values)
        if no_underflow_count == 0:
            self.logger.warning('Could not determine throughput burst: All data points raised an underflow exception')
        elif not underflow_flag:
            self.logger.warning('Could not determine throughput burst: No data points raised an underflow exception')
        else:
            # Store result in system dataset
            self.set_dataset_sys(self.THROUGHPUT_BURST_KEY, last_num_events)

    @rpc(flags={"async"})
    def _message_current_period(self, current_period):
        # Message current period
        self.logger.info('Using period {:s}'.format(dax.util.units.time_to_str(current_period)))

    @kernel
    def _calibrate_throughput_burst(self, num_events_scan, num_samples, current_period, period_step,
                                    no_underflow_cutoff, num_step_cutoff):
        # Storage for last number of events
        last_num_events = 0
        # Count of last number of events without underflow
        no_underflow_count = 0
        # A flag to mark if at least one underflow happened
        underflow_flag = False

        while True:
            # Reset variables
            last_num_events = 0
            underflow_flag = False
            no_underflow_count = 0

            # Iterate over scan
            for num_events in num_events_scan:
                try:
                    # Spawn events
                    self._spawn_events(current_period, num_samples, num_events)
                except RTIOUnderflow:
                    # Set underflow flag
                    underflow_flag = True
                    # Reset no underflow counter
                    no_underflow_count = 0
                else:
                    if no_underflow_count == 0:
                        # Store the number that works
                        last_num_events = num_events

                    # Increment counter
                    no_underflow_count += 1

                    if no_underflow_count >= no_underflow_cutoff:
                        # No underflow detected and cutoff reached, stop testing
                        break

            if num_step_cutoff == 0:
                break  # Max number of steps has been reached, stop testing in any case
            if not underflow_flag:
                # No underflow events occurred, reducing period
                current_period -= period_step
                num_step_cutoff -= 1
                self._message_current_period(current_period)
            elif no_underflow_count == 0:
                # All points had an underflow event, increasing period
                current_period += period_step
                num_step_cutoff -= 1
                self._message_current_period(current_period)
            else:
                break  # Underflow events happened and threshold was found, stop testing

        # Store results in dataset
        self.set_dataset('no_underflow_count', no_underflow_count)
        self.set_dataset('underflow_flag', underflow_flag)
        self.set_dataset('last_num_events', last_num_events)
        self.set_dataset('last_period', current_period)
        self.set_dataset('last_num_step_cutoff', num_step_cutoff)


class RtioLoopStressModule(RtioStressModule):
    """Module to stress the RTIO system with a looped connection."""

    # System keys
    LATENCY_RTIO_RTIO = 'latency_rtio_rtio'
    LATENCY_CORE_RTIO = 'latency_core_rtio'
    LATENCY_RTIO_CORE = 'latency_rtio_core'
    LATENCY_RTT = 'latency_rtt'  # Round-trip-time from RTIO input to RTIO output

    def build(self, ttl_out, ttl_in):
        # Call super
        super(RtioLoopStressModule, self).build(ttl_out)
        # TTL input device
        self.setattr_device(ttl_in, 'ttl_in')

    def load_module(self):
        # Call super
        super(RtioLoopStressModule, self).load_module()

        # Load latency parameters
        self.setattr_dataset_sys(self.LATENCY_RTIO_RTIO)
        self.setattr_dataset_sys(self.LATENCY_CORE_RTIO)
        self.setattr_dataset_sys(self.LATENCY_RTIO_CORE)
        self.setattr_dataset_sys(self.LATENCY_RTT)

    @kernel
    def init_module(self):
        # Call super (not using MRO/super because it is incompatible with the compiler, call parent function directly)
        RtioStressModule.init_module(self)

        # Break realtime to get some slack
        self.core.break_realtime()

        # Set TTL direction
        self.ttl_in.input()

    """Calibrate latency core-RTIO"""

    def calibrate_latency_core_rtio(self, num_samples):
        pass  # TODO

    """Calibrate latency RTIO-core"""

    def calibrate_latency_rtio_core(self, num_samples, detection_window):
        # Store input values in dataset
        self.set_dataset('num_samples', num_samples)
        self.set_dataset('detection_window', detection_window)

        # Prepare datasets for results
        self.set_dataset('t_zero', [])
        self.set_dataset('t_rtio', [])
        self.set_dataset('t_return', [])

        # Call the kernel
        self._calibrate_latency_rtio_core(num_samples, detection_window)

        # Get results
        t_zero = np.array(self.get_dataset('t_zero'))
        t_rtio = np.array(self.get_dataset('t_rtio'))
        t_return = np.array(self.get_dataset('t_return'))

        # Process results directly (next experiment might need these values)
        rtio_rtio = (t_rtio - t_zero).mean()
        rtio_core = (t_return - t_zero).mean()
        self.set_dataset_sys(self.LATENCY_RTIO_RTIO, rtio_rtio)
        self.set_dataset_sys(self.LATENCY_RTIO_CORE, rtio_core)

    @kernel
    def _calibrate_latency_rtio_core(self, num_samples, detection_window):
        # Reset core
        self.core.reset()

        for i in range(num_samples):
            # Turn output off
            self.core.break_realtime()  # Break realtime to prevent underflow exceptions
            self.ttl_out.off()
            delay(1 * us)  # Guarantee a delay between off and on

            # Guarantee a healthy amount of slack to start the measurement
            self.core.break_realtime()

            # Save time zero
            t_zero = now_mu()
            # Turn output on
            self.ttl_out.on()
            # Get the timestamp when the RTIO core detects the input event
            t_rtio = self.ttl_in.timestamp_mu(self.ttl_in.gate_rising(detection_window))
            # Get the timestamp (of the RTIO core) when the RISC core reads the input event (return time)
            t_return = self.core.get_rtio_counter_mu()  # Returns a lower bound

            # Store values at a non-critical time
            self.append_to_dataset_sys('t_zero', t_zero)
            self.append_to_dataset_sys('t_rtio', t_rtio)
            self.append_to_dataset_sys('t_return', t_return)

    """Calibrate RTT RTIO-core-RTIO"""

    def calibrate_latency_rtt(self, num_samples):
        pass  # TODO
