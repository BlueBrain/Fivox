Line events
===========

At the moment, Fivox works only with point events, which are also used as a very
simplistic approximation of elongated sources, e.g. LFP computation on neuron
structures (line-shaped). This is a very basic approach and not enough for
certain use cases. We need to extend the functionality in order to support more
types of events, such as lines.

## Requirements

* In addition to the current point events (we want to keep them to represent for
  punctual structures), we need to add a new type of events, such as lines, to
  represent elongated structures.
* A line event must be represented by: two 3D points, a radius and a value.
* We need the new events to be compatible with the existing ones. For example,
  we could evaluate different types of events in the same use case, e.g. somas
  (points) and dendrites (lines/cylinders).

## Implementation

The current Event class could serve as a base class and two new classes can
derive from it: PointEvent and LineEvent. The base class will hold the common
parts, such as the value of the event.

Then, each loader will create their own events depending on the use case
(points and/or lines), and the functors will evaluate all the events as if they
were base events.
The specific behaviour for each of the derived events can be implemented using:

* Templated functions in the functor
* Checking the class type through dynamic casting
* Adding methods in the event class that will be implemented in the derived ones
  (lineEvent, pointEvent, ...), doing whatever is needed depending on the type
* ... more alternatives to be discussed

## Issues

### 1: Use only one value for the radius or two different values (beginning and
end of line)?

Resolved: Use only one radius

We can use a single radius for the whole line, resulting in a cylinder instead
of a conic structure. Using two different radii would only increase the
complexity when evaluating the event, and the difference in the result might not
be too noticeable.

### 2: To create and update the rtree we still need to evaluate the positions of
the events. If, depending on the event type, we have more than one position,
which one do we use?

Resolved: Add a virtual method in Event to compute and return only one position

For the PointEvent objects the new method would directly return the event
position, but for LineEvents it needs to consider both ends of the line and
return the center of it (for example). This would be enough for its usage in
structures like the rtree in the EventSource, but when it comes to the
evaluation of the events to compute the content of the volume in each of the
functors, it still needs to consider both ends of the line (two positions).
