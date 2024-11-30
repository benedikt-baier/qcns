#pragma once

#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <tuple>

#include "qubit.h"
#include "errors.h"

class QChannel
{
    /*
    Represents a quantum channel

    Members:
        mutex (std::mutex): mutex used for synchronizing threads
        condition (std::condition_variable): condition used for synchronizing threads
        _queue (std::queue<std::pair<uint32_t, QSystem*>>): queue representing the transmission of qubits
        _errors (std::vector<Error*>): errors which are applied to each qubit traversing the channel
    */

public:
    double _length;
    double _signal_time;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::queue<std::tuple<double, uint32_t, QSystem*>> _queue;

    std::vector<Error*> _errors;

    QChannel(double length=0.0, const std::vector<Error*>& errors = {})
    {
        /*
        Initialises a quantum channel

        Args:
            errors (std::vector<Error*>): errors which are applied to each qubit traversing the channel
        */

        _length = length;
        _signal_time = _length / (3e5);
        _errors = errors;
    }

    void
    put(Qubit* q, double& host_time)
    {
        /*
        Puts a qubit into the channel

        Args:
            q (Qubit): qubit which is transmitted
            host_time (double): time of host

        Returns:
            /
        */

        std::lock_guard<std::mutex> lock(mutex);
        double arrival_time = host_time + _signal_time;
        std::tuple<double, uint32_t, QSystem*> encode = std::make_tuple(arrival_time, q->_index, q->_qsystem);
        _queue.push(encode);
        condition.notify_one();
    }

    std::tuple<double, Qubit*>
    get()
    {
        /*
        Retrives the first qubit from the channel

        Args:
            /

        Returns:
            q (Qubit): transmitted qubit
        */

        std::unique_lock<std::mutex> lock(mutex);
        while(_queue.empty())
        {
            condition.wait(lock);
        }

        std::tuple<double, uint32_t, QSystem*> t = _queue.front();
        _queue.pop();

        Qubit* q;
        double arrival_time = std::get<0>(t);
        q->_index = std::get<1>(t);
        q->_qsystem = std::get<2>(t);

        for (uint32_t i = 0; i < _errors.size(); i++)
            _errors[i]->apply(q);

        return std::make_tuple(arrival_time, q);
    }

    std::tuple<double, Qubit*>
    get_wait(uint32_t timeout)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
        if(_queue.empty())
            return std::make_tuple(0.0, nullptr);

        std::tuple<double, uint32_t, QSystem*> t = _queue.front();
        _queue.pop();

        Qubit* q;
        double arrival_time = std::get<0>(t);
        q->_index = std::get<1>(t);
        q->_qsystem = std::get<2>(t);

        for (uint32_t i = 0; i < _errors.size(); i++)
            _errors[i]->apply(q);

        return std::make_tuple(arrival_time, q);
    }
};

class CChannel
{
    /*
    Represents a classical channel

    Members:
        mutex (std::mutex): mutex used for synchronizing threads
        condition (std::condition_variable): condition used for synchronizing threads
        _queue (std::queue<std::string>): queue representing the transmission of classical messages

    */

public:
    double _length;
    double _signal_time;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::queue<std::tuple<double, std::string>> _queue;

    CChannel(double length=0.0)
    {
        /*
        Initialises a quantum channel

        Args:
            /
        */

        _length = length;
        _signal_time = _length / (3e5);
    }

    void
    put(std::string& message, double& host_time)
    {
        /*
        Puts a message in the channel

        Args:
            message (std::string): message to transmit

        Returns:
            /
        */

        std::lock_guard<std::mutex> lock(mutex);
        double arrival_time = host_time + _signal_time;
        _queue.push(std::make_tuple(arrival_time, message));
        condition.notify_one();
    }

    std::tuple<double, std::string>
    get()
    {
        /*
        Retrieves the first message from the channel

        Args:
            /

        Returns:
            message (std::string): transmitted message
        */

        std::unique_lock<std::mutex> lock(mutex);
        while (_queue.empty())
        {
            condition.wait(lock);
        }
        std::tuple<double, std::string> t = _queue.front();
        _queue.pop();

        return t;
    }

    std::tuple<double, std::string>
    get_wait(uint32_t timeout)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
        if(_queue.empty())
            return std::make_tuple(0.0, nullptr);

        std::tuple<double, std::string> t = _queue.front();
        _queue.pop();

        return t;
    }
};

