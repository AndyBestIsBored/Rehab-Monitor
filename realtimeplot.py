import socket
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Define the IP address and port number
IP_ADDRESS = "localhost"
PORT = 5005

# Define the figure and the 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the plot with empty data
line, = ax.plot([], [], [], lw=2)

# Set the axis limits
ax.set_xlim(-1000, 0)
ax.set_ylim(-500, 0)
ax.set_zlim(-300, 0)

# Create the socket connection
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((IP_ADDRESS, PORT))

# Define the animation function
def animate(i):
    # Receive the data from the socket
    data, _ = sock.recvfrom(1024)
    data = data.decode().strip().split(',')
    if len(data) == 3:
        # Convert the data to floats
        x, y, z = [float(d) for d in data]
        
        # Set the new data for the plot
        line.set_data(np.append(line.get_xdata(), x), np.append(line.get_ydata(), y))
        line.set_3d_properties(np.append(line.get_data(), z))
    
    # Return the plot object
    return line,

# Animate the plot
ani = animation.FuncAnimation(fig, animate, interval=100, blit=True)

# Show the plot
plt.show()
