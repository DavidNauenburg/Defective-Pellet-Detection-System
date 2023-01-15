#!/usr/bin/env python3

# import necessary packages
import os
import plotly
from skimage import filters, measure, morphology, io
from skimage.color import rgb2gray
import plotly.express as px
import plotly.graph_objects as go
import math
import pandas as pd
import datetime

PASSED = "All parameters PASSED. You have a Perfect Pellet!<br>"
AREA_LARGE = "Failed: Area is greater than acceptable range<br>"
AREA_SMALL = "Failed: Area is less than acceptable range<br>"
CIRC_FAIL = "Failed: Eccentricity is greater than acceptable ratio<br>" 
PERM_LARGE = "Failed: Perimeter is greater than acceptable range<b>"
PERM_SMALL = "Failed: Perimeter is less than acceptable range<br>"
DIAM_LARGE = "Failed: Diameter is greater than acceptable range<br>"
DIAM_SMALL = "Failed: Diameter is less than acceptable range<br>"

#diam_file = open("diameters.txt", "a")

class ImageAnalysis():
    #def __init__(self, image_path = "")
    def __init__(self, image_path, ref_width, counter):

        # Save path to image to variable path_to_image
        self.image_received = datetime.datetime.now()
        #self.path_to_image = image_path
        #self.path_to_image = r"10May_17ul_2.80mm_pellet_ref.jpg"

        self.path_to_image = image_path
        #self.image = image

        # Width of reference object
        # TODO: this value needs to be set by the user
        #self.ref_width = 3.175 #20.64
        self.ref_width = ref_width

        # Used to convert pixel-to-metric
        self.pixelsPerMetric = None

        ################################################################
        # These variables are attributes of each contour that are used 
        # to determine whether a pellet passes or fails quality control
        self.diameter = 0.0
        self.area = 0.0
        self.perimeter = 0.0
        self.eccentricity = 0.0

        ################################################################
        # Initial image processing
        # Read image from given path and convert to grayscale
        self.image = io.imread(self.path_to_image, as_gray=False)

        self.gray_image = rgb2gray(self.image)

        # Reduce noise with Gaussian filter
        self.blurred_image = filters.gaussian(
            self.gray_image, 
            sigma=1.0, 
            multichannel=False #channel_axis
        )

        # Binary image, post-process the binary mask and compute labels
        self.threshold = filters.threshold_otsu(self.blurred_image)
        self.mask = self.gray_image > self.threshold
        self.mask = morphology.remove_small_objects(self.mask, 50)
        self.mask = morphology.remove_small_holes(self.mask, 50)

        ################################################################

        # These labels determine which pixels are connected to each 
        # other 
        # The contours of each object will be determined from these 
        # labels
        self.labels = measure.label(self.mask)

        # The scatter plot will be traced over the original image with 
        # hover labels that are visible when the cursor is over each 
        # highlighted contour
        self.fig = px.imshow(self.image, binary_string=False)
        self.fig.update_traces(hoverinfo='skip')

        # Determines properties (attributes) for each label in a given 
        # image
        self.props = measure.regionprops(self.labels, self.gray_image)

        # Can access more attributes than the 4 listed below including
        # coords and centroid
        '''
        self.properties = [
            'area',
            'eccentricity',
            'perimeter',
            'mean_intensity'
        ]
        

        self.pellet_data = {
            self.properties[0]: [],
            self.properties[1]: [],
            self.properties[2]: [],
            self.properties[3]: [],
            'diameter': [],
            'centroid': []
        }
        '''
        self.properties = [
            'area',
            'eccentricity',
            'mean_intensity'
        ]
        self.pellet_data = {
            self.properties[0]: [],
            self.properties[1]: [],
            self.properties[2]: [],
            'diameter': [],
            'centroid': []
        }

        ################################################################

        diam = 0
        count = 0
        # Calculate the value of each parameter for each contour
        for index in range(0, self.labels.max()):
            self.begin_time = datetime.datetime.now()

            self.label_i = self.props[index].label
            self.contour = measure.find_contours(
                self.labels == self.label_i, 0.5
            )[0]

            self.y, self.x = self.contour.T

            # Create the hover label for each contour
            self.hoverinfo = ''

            # Iterate through each attribute (property) of the contour
            for prop_name in self.properties:

                self.attr = getattr(self.props[index], prop_name)

                if prop_name == "area":

                    self.diameter = 2 * (math.sqrt((self.attr/(math.pi))))

                    if self.pixelsPerMetric is None:
                        self.pixelsPerMetric = self.diameter / self.ref_width

                    self.attr = self.attr / (self.pixelsPerMetric**2)
                    self.area = self.attr
                '''
                elif prop_name == "perimeter":
                    pass
                    self.attr = self.attr / self.pixelsPerMetric
                    self.perimeter = self.attr

                elif prop_name == "eccentricity":
                    self.eccentricity = self.attr
                '''
                if prop_name == 'eccentricity':
                    self.eccentricity = self.attr 

                self.pellet_data[prop_name].append(self.attr)

                # Concatenate the info of each attribute into the 
                # hoverinfo label that the user will see in the image
                self.hoverinfo += f'<b>{prop_name}: {self.attr:.3f}</b><br>'

            self.diameter = self.diameter / self.pixelsPerMetric
            self.pellet_data['diameter'].append(self.diameter)
            diam += self.diameter
            count += 1
            #diam_file.write(f"{self.diameter:.2f}\n")

            self.centroid = getattr(self.props[index], 'centroid')
            # Set the reference object to have coordinates (0,0)
            if index == 0:
                self.x_diff = self.centroid[1]
                self.y_diff = self.centroid[0]

            # Adjust the coordinates of each pellet based on the
            # reference
            self.pellet_data['centroid'].append(
                (round(self.centroid[1]-self.x_diff, 2),
                round(self.centroid[0]-self.y_diff, 2)
                )
            )

            self.hoverinfo += f'<b>{"diameter"}: {self.diameter:.2f}</b><br>'

            if (self.diameter >= 2.74
                and self.diameter <= 3.03
                and self.eccentricity <= 0.40):

                '''
                if (self.area <= 7.21
                    and self.area >= 5.89
                    and self.perimeter <= 9.52
                    and self.perimeter >= 8.60
                    and self.diameter >= 2.74
                    and self.diameter <= 3.03
                    and self.eccentricity <= 0.40):
                '''

                self.red = 0
                self.green = 255
                self.hoverinfo += PASSED
            else:
                self.red = 255
                self.green = 0

            '''
            if self.area > 7.21:
                self.hoverinfo += AREA_LARGE
            if self.area < 5.89:
                self.hoverinfo += AREA_SMALL
            '''
            if self.eccentricity > 0.40:
                self.hoverinfo += CIRC_FAIL
            '''
            if self.perimeter > 9.52:
                self.hoverinfo += PERM_LARGE
            if self.perimeter < 8.60:
                self.hoverinfo += PERM_SMALL
            '''
            if self.diameter > 3.03:
                self.hoverinfo += DIAM_LARGE
            if self.diameter < 2.74:
                self.hoverinfo += DIAM_SMALL

            # Highlight each contour green or red depending on whether 
            # or not they pass or fail each parameter
            self.fig.add_trace(go.Scatter(
                x = self.x,  
                y = self.y,  
                name = "Reference" if self.label_i == 1 else (self.label_i-1),
                mode = 'lines',
                fill = 'toself',
                fillcolor = 'rgba(%s, %s, 0, 0.5)' % (self.red, self.green),
                showlegend = True,
                hovertemplate = self.hoverinfo,
                hoveron = 'points+fills',
                marker = dict(color='rgba(%s,%s,0,0.5)'%(self.red, self.green))
                )
            )

            self.end_time = datetime.datetime.now()
            print("Time it took to calculate every parameter and highlight each contour: %s" % (
                self.end_time - self.begin_time))
        avg = diam / count
        print(f'{"Average Diameter"}: {avg:.2f}')

        ################################################################
        # Show figure with all the highlighted contours over each object
        # in the original image
        # FIXME: uncomment to display interactive image with labels

        #diam_file.close()

        #plotly.io.show(self.fig)
        temp = os.path.dirname(image_path)
        output = temp + f"\output_fig_{counter}.html"
        self.fig.write_html(output)

        ################################################################
        # Show all the information from the hover labels in a table
        self.df = pd.DataFrame(data=self.pellet_data)
        self.df = self.df.rename(
            index=lambda x: 'Reference' if x == 0 else ('Pellet %s' % (x))
        )
        # Make the index a column in the dataframe
        self.df = self.df.rename_axis('Pellet Index').reset_index()
        self.data_displayed = datetime.datetime.now()

        print("Time it took from image recieved to data displayed: %s" % (self.data_displayed - self.image_received))
        return

# def main(image_path)
def main():
    #image_path = r"9May_320_test1.jpg"
    #image_path = r"10May_19ul_2.97mm_ref.jpg"
    analysis = ImageAnalysis()

#if __name__ == '__main__':
    #main()
