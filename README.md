# Juggle Counter

Side project using OpenCV to count the number of throws while juggling.

Almost all existing projects of this nature depend on either filtering for a specific ball colour (e.g. green tennis balls), or using a ball of a specific (pre-determined) size.

You should be able to juggle with whatever balls you wish.

## Ball detection

Initially used Hough Circle transform to detect circles. However, this required parameter tweaking that I couldn't consistently get working. In addition, it would require a background with few distractions (even non-circular objects will be detected by the transform).

Current iteration is using background subtraction to detect balls moving. The pixels that are found via background subtraction are eroded and dilated to remove inconsistencies. Countours are then found around the remaining pixels.

## Counting throws

A buffer of ball centers is kept. Comparing a position several steps ago to the most current step allows one to determine the direction the ball is moving. When a ball moves from 'upwards' to 'downwards' the throw count is incremented.

A buffer is used as there are inconsistences when using only the previous position (too much natural fluctuation in position).
