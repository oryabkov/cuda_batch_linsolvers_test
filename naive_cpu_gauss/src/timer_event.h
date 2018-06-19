#ifndef __TIMER_EVENT_H__
#define __TIMER_EVENT_H__

#include <stdexcept>

struct timer_event
{
    virtual void    init() = 0;
    virtual void    record() = 0;
    virtual double  elapsed_time(const timer_event &e0)const = 0;
    virtual void    release() = 0;
};

#endif
