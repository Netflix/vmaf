fix(hip): wire `motion_max_val` clip into `float_motion_hip` collect and flush paths

`float_motion_hip` applied `motion_fps_weight` but never clipped the result
against `motion_max_val`, causing the option to be silently ignored on the HIP
backend. Clips `motion2` and the flush tail score to `motion_max_val` after
applying `fps_weight`, matching the CPU `float_motion.c` behaviour at lines 514
and 342. Identity at the default value of 10000.0.
