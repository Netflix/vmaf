/**
 *
 *  Copyright 2016-2017 Netflix, Inc.
 *
 *     Licensed under the Apache License, Version 2.0 (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>

class Timer {
    typedef std::chrono::high_resolution_clock hrclock;

    hrclock::time_point m_start;
    hrclock::time_point m_stop;
public:
    void start() { m_start = hrclock::now(); }

    void stop() { m_stop = hrclock::now(); }

    double elapsed()
    {
        std::chrono::duration<double> secs = m_stop - m_start;
        return secs.count();
    }
};

#endif // TIMER_H_
