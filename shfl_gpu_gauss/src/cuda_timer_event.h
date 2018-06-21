#ifndef _CUDA_TIMER_EVENT_H__
#define _CUDA_TIMER_EVENT_H__

#include "cuda_safe_call.h"
#include "timer_event.h"

struct cuda_timer_event : public timer_event
{
        cudaEvent_t     e;

        cuda_timer_event()
        {
        }
        ~cuda_timer_event()
        {
        }
        virtual void    init()
        {
                CUDA_SAFE_CALL( cudaEventCreate( &e ) );
        }
        virtual void    record()
        {
                cudaEventRecord( e, 0 );

        }
        virtual double  elapsed_time(const timer_event &e0)const
        {
                const cuda_timer_event *cuda_event = dynamic_cast<const cuda_timer_event*>(&e0);
                if (cuda_event == NULL) {
                        throw std::logic_error("cuda_timer_event::elapsed_time: try to calc time from different type of timer (non-cuda)");
                }
                float   res;
                cudaEventSynchronize( e );
                cudaEventElapsedTime( &res, cuda_event->e, e );
                return (double)res;
        };
        virtual void    release()
        {
                cudaEventDestroy( e );
        }
};

#endif
