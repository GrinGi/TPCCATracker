/*
 * This file is part of TPCCATracker package
 * Copyright (C) 2007-2020 FIAS Frankfurt Institute for Advanced Studies
 *               2007-2020 Goethe University of Frankfurt
 *               2007-2020 Ivan Kisel <I.Kisel@compeng.uni-frankfurt.de>
 *               2007-2019 Sergey Gorbunov
 *               2007-2019 Maksym Zyzak
 *               2007-2014 Igor Kulakov
 *               2014-2020 Grigory Kozlov
 *
 * TPCCATracker is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TPCCATracker is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef ALIHLTTPCCADISPLAY_H
#define ALIHLTTPCCADISPLAY_H

#include "AliHLTTPCCAParameters.h"
#include "AliHLTArray.h"

class AliHLTTPCCAGBTracker;
class AliHLTTPCCATracker;
class AliHLTTPCCATrack;
class AliHLTTPCCATrackParam;
class AliHLTTPCCAParam;
class AliHLTTPCCAPerformance;
class AliHLTTPCCAHitLabel;
class AliHLTTPCCAMCTrack;
class AliHLTTPCCALocalMCPoint;
class TCanvas;
class TPad;
#include "TArc.h"
#include "TLine.h"
#include "TPolyLine.h"
#include "TBox.h"
#include "TCrown.h"
#include "TMarker.h"
#include "TLatex.h"
#include "TArrow.h"

#include "TWbox.h"

#include <vector>
using std::vector;


/**
 * @class AliHLTTPCCADisplay
 */
class AliHLTTPCCADisplay
{

  public:

    class AliHLTTPCCADisplayTmpHit;

    static AliHLTTPCCADisplay &Instance();

    AliHLTTPCCADisplay();

    virtual ~AliHLTTPCCADisplay();

    void Init();
    void Update();
    void ClearView();
    void ClearViewPT();
    void Ask();
    void SetSliceView();
    void SetTPCView();
    void SetTPC( const AliHLTTPCCAParam& tpcParam);  // iklm
    void SetCurrentSlice( AliHLTTPCCATracker *slice );
    void SetGB( const AliHLTTPCCAGBTracker * GBTracker );
    void Set2Slices( AliHLTTPCCATracker * slice );

    const AliHLTTPCCAGBTracker * GetGB(){return fGB;};
    int GetColor( int i ) const;
    int GetColorZ( double z ) const ;
    int GetColorY( double y ) const ;
    int GetColorK( double k ) const ;
    void Global2View( double x, double y, double *xv, double *yv ) const ;
    void Slice2View( double x, double y, double *xv, double *yv ) const ;
    void Slice2ViewPT( double x, double y, double z, double *phi, double *teta ) const ;
    int GetTrackMC( const AliHLTTPCCADisplayTmpHit *vHits, int NHits );
  

    void DrawTPC();
    void DrawSlice( AliHLTTPCCATracker *slice, bool DrawRows = 0, bool DrawGrid = 1 );
    // ---
#if 0
    void DrawSliceGridPT( AliHLTTPCCATracker *slice, int np = 10, int nt = 15 );
    void DrawRow5();
    void DrawHitsRow5( AliHLTTPCCATracker *slice, int r, int color = -1, Size_t width = -1 );
#endif
    // ---
///mvz start 20.01.2010
    void DrawPoint(float x, float y, float z, int Start = 1, float width = 1 );
    void DrawGBPoint(const AliHLTTPCCAGBTracker &tracker, int iSlice, float x, float y, float z, int Start = 1, Size_t width = 1 ); ///mvz 27.01.2010
    void DrawGBPoint( float x, float y, float z, int Start = 1, Size_t width = 1 ); ///mvz 27.01.2010
    void DrawGBPoint(float p[6], int Start, int color=kBlack, Size_t width = 1 );
    void DrawGBPoint(AliHLTTPCCATracker &slice, float x, float y, float z, int Start=1, Size_t width=1 ); ///mvz 29.01.2010
#ifdef DO_TPCCATRACKER_EFF_PERFORMANCE
    void DrawSliceOutTrackParam( int itr, int color, Size_t width ); ///mvz 03.02.2010
    void DrawSliceOutTrack1( int itr, int color, Size_t width ); ///mvz 03.02.2010
#endif
    void DrawHelix(float p0, float c, float z, float zStart, float z0, float xc, float yc, float r, float b, int color, Size_t width); ///mvz 10.02.2010
    void DrawParticleGlobal(float *param, float q, float tStart, float tEnd, float b, int color=kOrange, Size_t width = 1);
///mvz end 20.01.2010
    void DrawSliceOutTrack( int itr, int color = -1, Size_t width = -1  );
    void DrawSliceOutTrack( AliHLTTPCCATrackParam &t, double Alpha, int itr, int color = -1, Size_t width = -1  );
    //void DrawSliceTrack( int itr, int color = -1 );
    bool DrawTrack( AliHLTTPCCATrackParam t, double Alpha, const AliHLTTPCCADisplayTmpHit *vHits,
                    int NHits, int color = -1, Size_t width = -1, bool pPoint = 0 );

    void DrawGBTrack( int itr, int color = -1, int width = -1 );
    void DrawGBTrackFast( const AliHLTTPCCAGBTracker &tracker, int itr, int color = -1 );
    bool DrawTracklet( AliHLTTPCCATrackParam &track, const int *hitstore, int color = -1, int width = -1, bool pPoint = 0 );

//     void DrawGBLinks( const AliHLTTPCCAGBTracker &tracker, int color = -1, Size_t width = -1 );
    
    void DrawGBHit( const AliHLTTPCCAGBTracker &tracker, int iHit, int color = -1, Size_t width = -1 );
    void DrawGBHits( const AliHLTTPCCAGBTracker &tracker, int color = -1, Size_t width = -1, int hitsType = -1 );

    void DrawSliceHit( int iRow, int iHit, int color = -1, Size_t width = -1 );
    void DrawSliceHitPT( int iRow, int iHit, int color = -1, Size_t width = -1 );
    void DrawSliceHits( int color = -1, Size_t width = -1 );
    void DrawSliceHitsPT( int color = -1, Size_t width = -1 );
    void DrawSliceLinks( int colorUp = -1, int colorDn = -1, int width = -1 );
    void DrawSliceLinksPT( int colorUp = -1, int colorDn = -1, int width = -1 );
    void DrawSliceLink( int iRow, int iHit, int colorUp = -1, int colorDn = -1, int width = -1 );
    void DrawSliceLinkPT( int iRow, int iHit, int colorUp = -1, int colorDn = -1, int width = -1 );

    void DrawTetaTrack( float teta, int iSlice );
    void DrawTetaTrackMerged( float iTeta, float nTeta, float iSlice, float nSlice, bool ok = false, bool okmc = false );
    void DrawDzDsPtTrack( float dzds, float pt, bool ok = false, bool ok1 = false );


    // ---
    void DrawTrackFromMerger( std::vector<double> x, std::vector<double> y, std::vector<double> z, int iSlice, std::vector<int> row, int color = -1, Size_t width = -1 );
    void DrawTrackFromMergerPT( std::vector<double> x0, std::vector<double> y0, std::vector<double> z0, int iSlice0, std::vector<int> row0, std::vector<double> x1, std::vector<double> y1, std::vector<double> z1, int iSlice1, std::vector<int> row1, int color = -1, Size_t width = -1 );
    void DrawTrackFromMergerPT_test( double x0, double x1, double y0, double y1, double z0, double z1, int row0, int row1, int slice0, int slice1, int color = -1, Size_t width = -1 );
    void DrawSlicesForTeta();
    void DrawFieldDzDsP();

    void DrawWrongMerge( float x0, float x1, float y0, float y1 );
    void DrawCorrectMerge( float x0, float x1, float y0, float y1, bool ok = false );

#if 0
    void DrawSliceHitsTest( int color = -1, Size_t width = -1 );
#endif
    void DrawSliceLine( float x1, float y1, float z1, float x2, float y2, float z2, int color = kBlue, float width = 1. );
#if 0
    void DrawSliceLineUV( float x1, float y1, float z1, float x2, float y2, float z2, int color = kBlue, float width = 1. );
    void DrawSliceLineUVXZ( float x1, float u1, float v1, float z1, float x2, float u2, float v2, float z2, int color = kBlue, float width = 1. );
    void DrawSliceLineUVXZnoFabs( float x1, float u1, float v1, float z1, float x2, float u2, float v2, float z2, int color = kBlue, float width = 1. );
    void DrawNumber( float x, float u, float v, float z, int N, float width = 1. );
#endif
    void DrawCircle( float x, float y, float r, int color = kBlack, float width = -1 );
#if 0
    void DrawPV( float x, float y, float z );
    void DrawUVZero( float x, float y, float z );
#endif
    void DrawPointXYZ(float x, float y, float z, int color = -1, Size_t width = -1 );
#if 0
    void DrawNumber( float x, float y, float z, int num, int color = -1 );
    void InitUVXYtest( AliHLTTPCCATracker *slice, float y0 = -0.005, float x0 = -0.0001, float y1 = 0.005, float x1 = 0.017 );
//    void ClearUVXYtest();
#endif
    void DrawCircleUVXYtest( float x, float y, float r, int color = kBlack, float width = -1 );
    void DrawPointXZ(float x, float z, int color = -1, Size_t width = -1 );
#if 0
    void DrawPVUVXY( float x, float y, float u, float v );
#endif
    void DrawPointXZfromXY(float x, float z, int color = -1, Size_t width = -1 );
    void DrawCircleXZfromXY( float x, float y, float r, int color = kBlack, float width = -1 );
#if 0
    void DrawSliceLineUVXYtest( float x1, float y1, float u1, float v1, float x2, float y2, float u2, float v2, int color = kBlue, float width = 1. );
#endif
    void DrawRectangleXYZ( float x, float y, float z, float lx, float ly, float lz, int color = kBlack, int border = -1 );
#if 0
    void DrawSlicePixel( AliHLTTPCCATracker *slice, float step, bool DrawRows = 0, bool DrawGrid = 1 );
#endif
    void DrawSliceLine1( float x1, float y1, float z1, float x2, float y2, float z2, int color = kBlue, float width = 1. );
    void DrawPointXYZl( float x, float y, float z, int color = kBlack, Size_t width = 1. );

    void SetSliceUV();
    void DrawPointYX( float x, float y, int color = kBlue, float width = 1. );

    void DrawBadMCHits( AliHLTResizableArray<AliHLTTPCCAHitLabel>* hitLabels );

    void DrawPointGlobal( float x, float y, float z, int Start = kRed, float width = 0.5 );
    void SetSliceGlobal( AliHLTTPCCATracker * slice );
    void SpecDrawHitGlobal( float X, float Y, float Z, int iSlice, int color = kRed, float width = 0.5 );
    void DrawHitGlobal( float x, float y, float z, int color = kRed, float width = 0.5 );
    void SpecDrawLine( float x1, float y1, float z1, int slice1, float x2, float y2, float z2, int slice2, int color = kRed, float width = 0.5 );
    void SpecDrawLineZ( float x1, float y1, float z1, int slice1, float x2, float y2, float z2, int slice2, int color = kRed, float width = 0.5 );
    void SpecDrawLineG( float x1, float y1, float z1, float x2, float y2, float z2, int color = kRed, float width = 0.5 );
    // ---

#ifdef XXXX

    void DrawMergedHit( int iRow, int iHit, int color = -1 );

    void DrawTrack( const AliHLTTPCCATrack &track, int color = -1, bool DrawCells = 1 );
#endif // XXXX
    void DrawTrackletPoint( const AliHLTTPCCATrackParam &t, int color = -1 );
    void DrawTrackParam( AliHLTTPCCATrackParam t, int color = 1 );

    void SetSliceTransform( double alpha );
    void SetSliceTransform( AliHLTTPCCATracker *slice );
    void SaveCanvasToFile( TString fileName);
        
    TPad *CanvasYX() { return fYX; }
    TPad *CanvasZX() { return fZX; }

    // ---
    TPad *CanvasPT() { return fPT; }
    // ---


/// IKu some new functional. Supposed to be usefull.
  struct TDrawObject {
    TDrawObject();

    virtual void DrawLocal(int iSlice){iSlice++;}; // draw in local coor
    virtual void DrawGlobal(){};// draw in global coor
    
    void SetColor(int c);  // set color for this track
    void SetWidth(float w);

         // please add here transformation, if you use some ....
    static void Global2View( double x, double y, double *xv, double *yv);
    static void Slice2View( double x, double y, double *xv, double *yv, float alpha);
    static void Slice2View( double x, double y, double *xv, double *yv, int iSlice);
    static float GetAlpha(const AliHLTTPCCAGBTracker *gbTracker, int iSlice);
   protected:
    enum{ NON = -1000000000 };

    static AliHLTTPCCADisplay& fDisplay; // display instance
    int fColor; // line color
    float fWidth; // width or size    
  };
  
  struct TDrawHit: public TDrawObject {
    TDrawHit()
    : x(NON)
    , y(NON)
    , z(NON)
    , alpha(NON)
    , gx(NON)
    , gy(NON)
    , gz(NON)
    , slice(NON)
    , row (NON)
    {}
    bool CheckAlpha(); // check if alpha was set. true - was set, false - wasn't
    bool CheckLocal();
    bool CheckGlobal();
    void FillGBCoor(); // take local cooridates and fill global

    virtual void DrawLocal(int iSlice); // draw in local coor
    virtual void DrawGlobal();// draw in global coor
    
    // ---
    void DrawHitPT();
    // ---

      // can be used one of coor. set:
    float x, y, z;    // sector coor.
    float alpha; 
    float gx, gy, gz; // global coor.
    // ---
    int slice;
    int row;
    // ---
    
   private:
    TMarker& GetMarker();
  };
  
  struct TDrawTrack: public TDrawObject {
    TDrawTrack();
    void FillGBHitsCoor(); // take local cooridates and fill global of hits

    vector<TDrawHit> hits;

    virtual void DrawLocal(int iSlice); // draw in local coor
    virtual void DrawGlobal();// draw in global coor
    virtual void DrawGlobalZ();// draw in global coor
    void DrawHitsLocal(int iSlice);
    void DrawHitsGlobal();
    
    // ---
    void DrawGlobalPT();
    void DrawGlobalTpad();
    void DrawGlobalPpad();
    void DrawGlobalTpad2hit( int nhits0 );
    void DrawSlicesForTeta();
    void DrawFieldDzDsP();
    void DrawDrawTetaTrackInt( float iTeta, int iSlice, bool ok = false, int color = kGreen );
    void DrawDrawTetaTrackIntMerged( float iTeta, float nTeta, float iSlice, float nSlice, bool ok = false, int color = kBlue );
    void DrawWrongMerge( float x0, float x1, float y0, float y1 );
    void DrawCorrectMerge( float x0, float x1, float y0, float y1, bool ok = false );
    // ---

      // draw adjustment
    void SetHitsColor(int c);  // set color for all hits
    void SetHitsWidth(float w);
    void SetStatColor(int c); // set color for line for all tracks
    void SetStatWidth(float w);

   private:
    TPolyLine& GetPolyLine();
    
    static int fStatColor;  // used for all tracks if nonstatic members are not set
    static float fStatWidth;
  };


  void SpecDrawMCTrackPointsGlobal(AliHLTTPCCAMCTrack& mcTrack, AliHLTResizableArray<AliHLTTPCCALocalMCPoint>* mcPointsArray, int color = kRed, float width = 0.5);
  void SpecDrawMCTrackHitsGlobal(AliHLTTPCCAMCTrack& mcTrack, AliHLTResizableArray<AliHLTTPCCALocalMCPoint>* mcPointsArray, AliHLTResizableArray<AliHLTTPCCAHitLabel>* hitLabels, int color = kGreen, float width = 0.5);
  void SpecDrawMCTrackLocal(AliHLTTPCCAMCTrack& mcTrack, AliHLTResizableArray<AliHLTTPCCALocalMCPoint>* mcPointsArray, int iSlice);

  void SpecDrawHitsFromMCTrackLocal(int iMC, const AliHLTResizableArray<AliHLTTPCCAHitLabel>* hitLabels, int iSlice);

  void SpecDrawRecoTrackGlobal( int iTr, int color = kBlue, float width = 0.5);

  // ---
  void SpecDrawRecoTrackGlobalPT( int iTr, int color = kBlue, float width = 0.5 );
  void SpecDrawMCTrackGlobalPT( AliHLTTPCCAMCTrack& mcTrack, AliHLTResizableArray<AliHLTTPCCALocalMCPoint>* mcPointsArray, int color, float width );
#if 0
  void SpecDrawMCTrackGlobalPT1( AliHLTTPCCAMCTrack& mcTrack, AliHLTResizableArray<AliHLTTPCCALocalMCPoint>* mcPointsArray, int color, float width );
  void SpecDrawMCTrackGlobalDzDsQPt( AliHLTTPCCAMCTrack& mcTrack, AliHLTResizableArray<AliHLTTPCCALocalMCPoint>* mcPointsArray, int color, float width );
#endif
  void SaveCanvasToFilePT( TString fileName);
  // ---
 public:
/// ENDOF IKu

  void SetDrawType( int t ) { fDrawType = t; }
  int DrawType() const { return fDrawType; }
  
  protected:

  TCanvas *fCanvas;              // the canvas
  TPad *fYX, *fZX;               // two views
  bool fAsk;                      // flag to ask for the pressing key
  bool fSliceView;               // switch between slice/TPC zoom
  AliHLTTPCCATracker *fSlice;      // current CA tracker, includes slice geometry
  const AliHLTTPCCAGBTracker *fGB;      // the Aliglobal tracker
  AliHLTTPCCAPerformance *fPerf; // Performance class (mc labels etc)
  double fCos, fSin, fZMin, fZMax, fYMin, fYMax;// view parameters
  double fSliceCos, fSliceSin;        // current slice angle
  double fRInnerMin, fRInnerMax, fROuterMin, fROuterMax, fTPCZMin, fTPCZMax; // view parameters

  TArc fArc;       // parameters of drawing objects are copied from this members
  TLine fLine;     //!
  TPolyLine fPLine;//!
  TMarker fMarker; //!
  TBox fBox;       //!
  TCrown fCrown;   //!
  TLatex fLatex;   //!
  TArrow fArrow;

  bool fDrawOnlyRef; // draw only clusters from ref. mc tracks

  int fDrawType; // what to draw, used from tracker. Values 1,2,3 is used.

  // --- For PhiTeta tests ---
  TCanvas *fCanvasPT;
  TPad *fPT;
  // ---

  private:
    AliHLTTPCCADisplay( const AliHLTTPCCADisplay& );
    AliHLTTPCCADisplay& operator=( const AliHLTTPCCADisplay& );
};

#endif
