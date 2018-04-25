nj = require 'numjs'

u = new Utils()
u.loadImageToCanvas("./img.jpg","canvasInput")

cannyEdgeDetection = () =>
  src = cv.imread('canvasInput')
  dst = new cv.Mat()
  cv.cvtColor(src, src, cv.COLOR_RGB2GRAY, 0)
  cv.Canny(src, dst, 50, 100, 3, false)
  cv.imshow('cannyOutput', dst)
  src.delete(); dst.delete()

contourDetection = () =>
  src = cv.imread('cannyOutput')
  dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC3) # Note had to switch rows/cols order here
  cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0)
  cv.threshold(src, src, 120, 200, cv.THRESH_BINARY)
  contours = new cv.MatVector()
  hierarchy = new cv.Mat()
  cv.findContours(src, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

  screenCnt = null

  for i in [0..contours.size()-1]
    c = contours.get(i)
    perimeter = cv.arcLength(c, true)
    approx = cv.Mat.zeros(src.cols, src.rows, cv.CV_8UC3)
    cv.approxPolyDP(c, approx, 0.02 * perimeter, true)

    if (cv.contourArea(approx) > 1000)
      screenCnt = approx
      matVec = new cv.MatVector()
      matVec.push_back(approx)
      cv.drawContours(dst, matVec, -1, new cv.Scalar(0,255,0), 2)

  cv.imshow('contourOutput', dst)
  src.delete(); dst.delete(); contours.delete(); hierarchy.delete()
  return screenCnt

orderPoints = (pts) =>
  console.log pts
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
  ###
	rect = nj.zeros([4, 2], dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[nj.argmin(s)]
	rect[2] = pts[nj.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

###
u.loadOpenCv  =>
  cannyEdgeDetection()
  screenCnt = contourDetection()
  console.log screenCnt
  orderPoints(nj.reshape(screenCnt, [4,2]))

