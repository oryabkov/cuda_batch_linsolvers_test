#ifndef __SYSTEM_TIMER_EVENT_H__
#define __SYSTEM_TIMER_EVENT_H__

//TODO windows realization

#include <sys/time.h>
#include <unistd.h>
#include <stdexcept>
#include "timer_event.h"

struct system_timer_event : public timer_event
{
    struct timeval tv;

    system_timer_event()
    {
    }
    ~system_timer_event()
    {
    }
    virtual void    init()
    {
    }
    virtual void    record()
    {
        gettimeofday(&tv, NULL);
    }
    virtual double  elapsed_time(const timer_event &e0)const
    {
        const system_timer_event *event = dynamic_cast<const system_timer_event*>(&e0);
        if (event == NULL) {
            throw std::logic_error("system_timer_event::elapsed_time: try to calc time from different type of timer");
        }
        double  res;
        long    seconds, useconds; 
        seconds  = tv.tv_sec  - event->tv.tv_sec;
        useconds = tv.tv_usec - event->tv.tv_usec;
        res = seconds*1000. + useconds/1000.0;
        return res;
    };
    virtual void    release()
    {
    }
};

#endif
