# TC-Tracking

## Detection and tracking of tropical cyclones.

### Current files description

annotation_handler routine - initial attempt based on the old lon, lat to x, y formula

Annotation_new routine - corrected version of lat, lon conversion formula and the csv creator with all the parameters required by the coco dataset format

json_creator routine - get the json for training and validation, passing to the models.

### In progress

Running the routine with Detectron API.
Make the annotation routine flexible with the resolution (add the resolution parameter assertion, factor per resolution).

### Current pipeline

Original labels csv (1)-> Formatted csv (2)-> COCO based JSON

(1) Annotation_new routine
(2) json_creator routine
