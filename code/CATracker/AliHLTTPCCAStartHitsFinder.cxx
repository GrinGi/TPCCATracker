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

#include "AliHLTTPCCAStartHitsFinder.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTArray.h"
#include "AliHLTTPCCAMath.h"

#ifdef USE_TBB
#include <tbb/parallel_sort.h>
#endif //USE_TBB

//////////////////////////////////////////////////////////////////////////////
// depends on:
// - linkUp/linkDown data for the rows above and below the row to process
//
// changes atomically:
// - number of tracklets
//
// writes
// - tracklet start-hit ids
//
// only when all rows are done, is NTracklets known
//////////////////////////////////////////////////////////////////////////////

void AliHLTTPCCAStartHitsFinder::run( AliHLTTPCCATracker &tracker, SliceData &data, int iter )
{

//  Vc::vector<AliHLTTPCCAStartHitId>& startHits = tracker.TrackletStartHits();
  std::vector<AliHLTTPCCAStartHitId>& startHits = tracker.TrackletStartHits();

  //TODO parallel_for
  const int rowStep = AliHLTTPCCAParameters::RowStep;
  const int lastRow = tracker.Param().NRows() - rowStep*2;
  for ( int rowIndex = 0; rowIndex <= lastRow; ++rowIndex ) {
//std::cout<<">rowIndex: "<<rowIndex<<"\n";
#ifdef USE_TBB
    int hitsStartOffset = CAMath::AtomicAdd( tracker.NTracklets(), 0 );
#else //USE_TBB
    const int hitsStartOffset = *tracker.NTracklets(); // number of start hits from other jobs
#endif //USE_TBB
    //X   int_m leftMask( Vc::Zero );
    const AliHLTTPCCARow &row = data.Row( rowIndex );
    int startHitsCount = 0;


    // look through all the hits and look for
    const int numberOfHits = row.NHits();
    for ( int hitIndex = 0; hitIndex < numberOfHits; hitIndex += SimdSizeInt ) {
      const int_v hitIndexes = IndexesFromZeroInt/*( Vc::IndexesFromZero )*/ + hitIndex;
      int_m validHitsMask = hitIndexes < numberOfHits;
      int_v hitDataTemp;
      for( unsigned int ii = 0; ii < SimdSizeFloat; ii++ ) {
//      	hitDataTemp[ii] = data.HitDataIsUsed( row )[(unsigned int)hitIndexes[ii]];
#ifndef USE_VC
	hitDataTemp.insert(ii, data.HitDataIsUsed( row )[(size_t)hitIndexes[ii]]);
#else
        hitDataTemp[ii] = data.HitDataIsUsed( row )[(unsigned int)hitIndexes[ii]];
#endif
      }
      validHitsMask &= ( hitDataTemp == int_v( 0 ) );
      // hits that have a link up but none down == the start of a Track
      const int_v &middleHitIndexes = data.HitLinkUpData( row, hitIndex );
      validHitsMask &= ( data.HitLinkDownData( row, hitIndex ) < int_v( 0 ) ) && ( middleHitIndexes >= int_v( 0 ) );

      if ( !validHitsMask.isEmpty() ) { // start hit has been found

          // find the length
        int iRow = rowIndex + 1*rowStep;
        int nRows = 2;
        int_v upperHitIndexes = middleHitIndexes;
        for (;!validHitsMask.isEmpty() && nRows < AliHLTTPCCAParameters::NeighboursChainMinLength[iter];) {
          for( unsigned int i = 0; i < SimdSizeFloat; i++ ) {
            if( !validHitsMask[i] ) continue;
//            upperHitIndexes[i] = data.HitLinkUpData( data.Row( iRow ) )[(unsigned int)upperHitIndexes[i]];
#ifndef USE_VC
            upperHitIndexes.insert(i, data.HitLinkUpData( data.Row( iRow ))[(size_t)upperHitIndexes[i]]);
#else
            upperHitIndexes[i] = data.HitLinkUpData( data.Row( iRow ))[(unsigned int)upperHitIndexes[i]];
#endif
          }
          validHitsMask &= upperHitIndexes >= int_v( 0 );
          nRows++;
          iRow += rowStep;
        }
          // check if the length is enough
        int_m goodChains = validHitsMask;
        if ( !goodChains.isEmpty() ) { 
            // set all hits in the chain as used
          data.SetHitAsUsed( row, static_cast<uint_v>( hitIndexes ), goodChains );
          int iRow2 = rowIndex + 1*rowStep;
          uint_v nHits(0);
//          nHits(goodChains) = 2;
#ifndef USE_VC
          nHits = KFP::SIMD::select(goodChains, 2, nHits);
#else
          nHits(goodChains) = 2;
#endif
          AliHLTTPCCARow curRow2;
          int_v upperHitIndexes2 = middleHitIndexes;
          for (;!goodChains.isEmpty();) {
            curRow2 = data.Row( iRow2 );

            data.SetHitAsUsed( curRow2, static_cast<uint_v>( upperHitIndexes2 ), goodChains );
            for( unsigned int i = 0; i < SimdSizeFloat; i++ ) {
              if( !goodChains[i] ) continue;
#ifndef USE_VC
              upperHitIndexes2.insert(i, data.HitLinkUpData( curRow2 )[(unsigned int)upperHitIndexes2[i]]);//[i] = data.HitLinkUpData( curRow2 )[(unsigned int)upperHitIndexes2[i]];
#else
              upperHitIndexes2[i] = data.HitLinkUpData( curRow2 )[(unsigned int)upperHitIndexes2[i]];
#endif
            }
            goodChains &= upperHitIndexes2 >= int_v( 0 );
//            nHits(goodChains)++;
#ifndef USE_VC
            nHits = KFP::SIMD::select(goodChains, nHits + 1, nHits);
#else
            nHits(goodChains)++;
#endif
            iRow2 += rowStep;
          }

          for( unsigned int i = 0; i < SimdSizeInt; i++ ) {
            if(!validHitsMask[i]) continue;
            startHits[hitsStartOffset + startHitsCount++].Set( rowIndex, hitIndex + i, nHits[i] );
//std::cout<<" - start hit: "<<rowIndex<<", "<<hitIndex + i<<", "<<nHits[i]<<"\n";
          }

          //   // check free space
          // if ( ISUNLIKELY( startHitsCount >= kMaxStartHits ) ) { // TODO take in account stages kMax for one stage should be smaller
          //   break;
          // }
        } // if good Chains
        
      } // if start hit
    } // for iHit

#ifdef USE_TBB
    hitsStartOffset = CAMath::AtomicAdd( tracker.NTracklets(), startHitsCount ); // number of start hits from other jobs
#else
    *tracker.NTracklets() += startHitsCount;
#endif //USE_TBB
  } // for rowIndex

#ifdef USE_TBB
  tbb::parallel_sort( startHits, startHits + *tracker.NTracklets() );
#else //USE_TBB
  std::sort( &startHits[0], &startHits[0] + *tracker.NTracklets() );
#endif //USE_TBB
}
