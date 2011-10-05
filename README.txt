This is Gary's multi-modal perception/version of linemod.

//INCLUDES
#include "mmod_objects.h"  //For train and test
#include "mmod_color.h"    //For depth and color processing (yes, I should change the name)

//Instantiate stuff
mmod_objects Objs; //Object train and test.
colorwta  calcColor;	//Feature processing
depthwta  calcDepth;	//Feature processing

Mat colorfeat, depthfeat;  //To hold feature outputs. These will be CV_8UC1 images

vector<Mat> FeatModes; //List of images
vector<string> modesCD; //Names of modes (color and depth)
string SessionID,ObjectName;  //Fill these in.
int framenum;
float learn_thresh = 0.8; //Just a guess
float match_threshold = 0.85; //Total guess
float frac_overlap = 0.6; //the fraction of overlap between 2 above threshold feature's bounding box rectangles that constitutes "overlap"

//SET UP:
//Set up our modes (right now we have color and depth. Lets say we use that order: Color and Depth)
	modesCD.push_back("Color");
	modesCD.push_back("Depth");
	framenum = ...
	SessionID = "Session1";
	ObjectName = "Cake";

//GET OUR IMAGES IN:
... Mat::ColorRaw ... //BGR image of type CV_8UC3
... Mat::DepthRaw ... //Depth image of type CV_16UC1
... Mat::Mask ...     //Object mask of type CV_8UC1 or CV_8UC3

//PROCESS TO GET FEATURES
calcColor.computeColorWTA(ColorRaw, colorfeat, Mask);
calcDepth.computeDepthWTA(DepthRaw, depthfeat, Mask);
FeatModes.clear();
FeatModes.push_back(colorfeat);
FeatModes.push_back(depthfeat);

//LEARN A TEMPLATE (for now, it will slow down with each view learned).

	string sT1("ST1"), oT("T"), oX("X");
	int num_templ = Objs.learn_a_template(FeatModes,modesCD, Mask, 
	                                      SessionID, ObjectName, framenum, learn_thresh);

. . . 

//TEST (note that you can also match_all_objects_at_a_point(...):
... Mat::ColorRaw ... //BGR image of type CV_8UC3
... Mat::DepthRaw ... //Depth image of type CV_16UC1
... Mat::AttendMask ...     //Attention mask of type CV_8UC1.  Can be empty for whole image
calcColor.computeColorWTA(ColorRaw, colorfeat, Mask);
calcDepth.computeDepthWTA(DepthRaw, depthfeat, Mask);
FeatModes.clear();
FeatModes.push_back(colorfeat);
FeatModes.push_back(depthfeat);

int skipX = 2, skipY = 2;  //These control sparse testing of the feature images

int num_matches = Objs.match_all_objects(FeatModes,modesCD,AttendMask,
                        match_threshold,frac_overlap,skipX,skipY);                                     

//TO DISPLAY MATCHES (NON-MAX SUPPRESSED)
Objs.draw_matches(ColorRaw);	                                     