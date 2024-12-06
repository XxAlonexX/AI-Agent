/* TA-LIB Copyright (c) 1999-2007, Mario Fortier
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in
 *   the documentation and/or other materials provided with the
 *   distribution.
 *
 * - Neither name of author nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* List of contributors:
 *
 *  Initial  Name/description
 *  -------------------------------------------------------------------
 *  MF       Mario Fortier
 *
 *
 * Change history:
 *
 *  MMDDYY BY   Description
 *  -------------------------------------------------------------------
 *  010802 MF   Template creation.
 *  052603 MF   Adapt code to compile with .NET Managed C++
 *
 */

/**** START GENCODE SECTION 1 - DO NOT DELETE THIS LINE ****/
/* All code within this section is automatically
 * generated by gen_code. Any modification will be lost
 * next time gen_code is run.
 */
/* Generated */ 
/* Generated */ #if defined( _MANAGED )
/* Generated */    #include "TA-Lib-Core.h"
/* Generated */    #define TA_INTERNAL_ERROR(Id) (RetCode::InternalError)
/* Generated */    namespace TicTacTec { namespace TA { namespace Library {
/* Generated */ #elif defined( _JAVA )
/* Generated */    #include "ta_defs.h"
/* Generated */    #include "ta_java_defs.h"
/* Generated */    #define TA_INTERNAL_ERROR(Id) (RetCode.InternalError)
/* Generated */ #else
/* Generated */    #include <string.h>
/* Generated */    #include <math.h>
/* Generated */    #include "ta_func.h"
/* Generated */ #endif
/* Generated */ 
/* Generated */ #ifndef TA_UTILITY_H
/* Generated */    #include "ta_utility.h"
/* Generated */ #endif
/* Generated */ 
/* Generated */ #ifndef TA_MEMORY_H
/* Generated */    #include "ta_memory.h"
/* Generated */ #endif
/* Generated */ 
/* Generated */ #define TA_PREFIX(x) TA_##x
/* Generated */ #define INPUT_TYPE   double
/* Generated */ 
/* Generated */ #if defined( _MANAGED )
/* Generated */ int Core::MacdExtLookback( int           optInFastPeriod, /* From 2 to 100000 */
/* Generated */                          MAType        optInFastMAType,
/* Generated */                          int           optInSlowPeriod, /* From 2 to 100000 */
/* Generated */                          MAType        optInSlowMAType,
/* Generated */                          int           optInSignalPeriod, /* From 1 to 100000 */
/* Generated */                          MAType        optInSignalMAType ) /* Generated */ 
/* Generated */ #elif defined( _JAVA )
/* Generated */ public int macdExtLookback( int           optInFastPeriod, /* From 2 to 100000 */
/* Generated */                           MAType        optInFastMAType,
/* Generated */                           int           optInSlowPeriod, /* From 2 to 100000 */
/* Generated */                           MAType        optInSlowMAType,
/* Generated */                           int           optInSignalPeriod, /* From 1 to 100000 */
/* Generated */                           MAType        optInSignalMAType ) /* Generated */ 
/* Generated */ #else
/* Generated */ int TA_MACDEXT_Lookback( int           optInFastPeriod, /* From 2 to 100000 */
/* Generated */                        TA_MAType     optInFastMAType,
/* Generated */                        int           optInSlowPeriod, /* From 2 to 100000 */
/* Generated */                        TA_MAType     optInSlowMAType,
/* Generated */                        int           optInSignalPeriod, /* From 1 to 100000 */
/* Generated */                        TA_MAType     optInSignalMAType ) /* Generated */ 
/* Generated */ #endif
/**** END GENCODE SECTION 1 - DO NOT DELETE THIS LINE ****/
{
   /* insert local variable here */
   int tempInteger, lookbackLargest;

/**** START GENCODE SECTION 2 - DO NOT DELETE THIS LINE ****/
/* Generated */ #ifndef TA_FUNC_NO_RANGE_CHECK
/* Generated */    /* min/max are checked for optInFastPeriod. */
/* Generated */    if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
/* Generated */       optInFastPeriod = 12;
/* Generated */    else if( ((int)optInFastPeriod < 2) || ((int)optInFastPeriod > 100000) )
/* Generated */       return -1;
/* Generated */ 
/* Generated */    #if !defined(_MANAGED) && !defined(_JAVA)
/* Generated */    if( (int)optInFastMAType == TA_INTEGER_DEFAULT )
/* Generated */       optInFastMAType = (TA_MAType)0;
/* Generated */    else if( ((int)optInFastMAType < 0) || ((int)optInFastMAType > 8) )
/* Generated */       return -1;
/* Generated */ 
/* Generated */    #endif /* !defined(_MANAGED) && !defined(_JAVA)*/
/* Generated */    /* min/max are checked for optInSlowPeriod. */
/* Generated */    if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
/* Generated */       optInSlowPeriod = 26;
/* Generated */    else if( ((int)optInSlowPeriod < 2) || ((int)optInSlowPeriod > 100000) )
/* Generated */       return -1;
/* Generated */ 
/* Generated */    #if !defined(_MANAGED) && !defined(_JAVA)
/* Generated */    if( (int)optInSlowMAType == TA_INTEGER_DEFAULT )
/* Generated */       optInSlowMAType = (TA_MAType)0;
/* Generated */    else if( ((int)optInSlowMAType < 0) || ((int)optInSlowMAType > 8) )
/* Generated */       return -1;
/* Generated */ 
/* Generated */    #endif /* !defined(_MANAGED) && !defined(_JAVA)*/
/* Generated */    /* min/max are checked for optInSignalPeriod. */
/* Generated */    if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
/* Generated */       optInSignalPeriod = 9;
/* Generated */    else if( ((int)optInSignalPeriod < 1) || ((int)optInSignalPeriod > 100000) )
/* Generated */       return -1;
/* Generated */ 
/* Generated */    #if !defined(_MANAGED) && !defined(_JAVA)
/* Generated */    if( (int)optInSignalMAType == TA_INTEGER_DEFAULT )
/* Generated */       optInSignalMAType = (TA_MAType)0;
/* Generated */    else if( ((int)optInSignalMAType < 0) || ((int)optInSignalMAType > 8) )
/* Generated */       return -1;
/* Generated */ 
/* Generated */    #endif /* !defined(_MANAGED) && !defined(_JAVA)*/
/* Generated */ #endif /* TA_FUNC_NO_RANGE_CHECK */
/**** END GENCODE SECTION 2 - DO NOT DELETE THIS LINE ****/

   /* insert lookback code here. */

   /* Find the MA with the largest lookback */
   lookbackLargest = LOOKBACK_CALL(MA)( optInFastPeriod, optInFastMAType );
   tempInteger     = LOOKBACK_CALL(MA)( optInSlowPeriod, optInSlowMAType );
   if( tempInteger > lookbackLargest )
      lookbackLargest = tempInteger;

   /* Add to the largest MA lookback the signal line lookback */
   return lookbackLargest + LOOKBACK_CALL(MA)( optInSignalPeriod, optInSignalMAType );
}

/**** START GENCODE SECTION 3 - DO NOT DELETE THIS LINE ****/
/*
 * TA_MACDEXT - MACD with controllable MA type
 * 
 * Input  = double
 * Output = double, double, double
 * 
 * Optional Parameters
 * -------------------
 * optInFastPeriod:(From 2 to 100000)
 *    Number of period for the fast MA
 * 
 * optInFastMAType:
 *    Type of Moving Average for fast MA
 * 
 * optInSlowPeriod:(From 2 to 100000)
 *    Number of period for the slow MA
 * 
 * optInSlowMAType:
 *    Type of Moving Average for slow MA
 * 
 * optInSignalPeriod:(From 1 to 100000)
 *    Smoothing for the signal line (nb of period)
 * 
 * optInSignalMAType:
 *    Type of Moving Average for signal line
 * 
 * 
 */
/* Generated */ 
/* Generated */ #if defined( _MANAGED ) && defined( USE_SUBARRAY )
/* Generated */ enum class Core::RetCode Core::MacdExt( int    startIdx,
/* Generated */                                         int    endIdx,
/* Generated */                                         SubArray^    inReal,
/* Generated */                                         int           optInFastPeriod, /* From 2 to 100000 */
/* Generated */                                         MAType        optInFastMAType,
/* Generated */                                         int           optInSlowPeriod, /* From 2 to 100000 */
/* Generated */                                         MAType        optInSlowMAType,
/* Generated */                                         int           optInSignalPeriod, /* From 1 to 100000 */
/* Generated */                                         MAType        optInSignalMAType,
/* Generated */                                         [Out]int%    outBegIdx,
/* Generated */                                         [Out]int%    outNBElement,
/* Generated */                                         cli::array<double>^  outMACD,
/* Generated */                                         cli::array<double>^  outMACDSignal,
/* Generated */                                         cli::array<double>^  outMACDHist )
/* Generated */ #elif defined( _MANAGED )
/* Generated */ enum class Core::RetCode Core::MacdExt( int    startIdx,
/* Generated */                                         int    endIdx,
/* Generated */                                         cli::array<double>^ inReal,
/* Generated */                                         int           optInFastPeriod, /* From 2 to 100000 */
/* Generated */                                         MAType        optInFastMAType,
/* Generated */                                         int           optInSlowPeriod, /* From 2 to 100000 */
/* Generated */                                         MAType        optInSlowMAType,
/* Generated */                                         int           optInSignalPeriod, /* From 1 to 100000 */
/* Generated */                                         MAType        optInSignalMAType,
/* Generated */                                         [Out]int%    outBegIdx,
/* Generated */                                         [Out]int%    outNBElement,
/* Generated */                                         cli::array<double>^  outMACD,
/* Generated */                                         cli::array<double>^  outMACDSignal,
/* Generated */                                         cli::array<double>^  outMACDHist )
/* Generated */ #elif defined( _JAVA )
/* Generated */ public RetCode macdExt( int    startIdx,
/* Generated */                         int    endIdx,
/* Generated */                         double       inReal[],
/* Generated */                         int           optInFastPeriod, /* From 2 to 100000 */
/* Generated */                         MAType        optInFastMAType,
/* Generated */                         int           optInSlowPeriod, /* From 2 to 100000 */
/* Generated */                         MAType        optInSlowMAType,
/* Generated */                         int           optInSignalPeriod, /* From 1 to 100000 */
/* Generated */                         MAType        optInSignalMAType,
/* Generated */                         MInteger     outBegIdx,
/* Generated */                         MInteger     outNBElement,
/* Generated */                         double        outMACD[],
/* Generated */                         double        outMACDSignal[],
/* Generated */                         double        outMACDHist[] )
/* Generated */ #else
/* Generated */ TA_RetCode TA_MACDEXT( int    startIdx,
/* Generated */                        int    endIdx,
/* Generated */                        const double inReal[],
/* Generated */                        int           optInFastPeriod, /* From 2 to 100000 */
/* Generated */                        TA_MAType     optInFastMAType,
/* Generated */                        int           optInSlowPeriod, /* From 2 to 100000 */
/* Generated */                        TA_MAType     optInSlowMAType,
/* Generated */                        int           optInSignalPeriod, /* From 1 to 100000 */
/* Generated */                        TA_MAType     optInSignalMAType,
/* Generated */                        int          *outBegIdx,
/* Generated */                        int          *outNBElement,
/* Generated */                        double        outMACD[],
/* Generated */                        double        outMACDSignal[],
/* Generated */                        double        outMACDHist[] )
/* Generated */ #endif
/**** END GENCODE SECTION 3 - DO NOT DELETE THIS LINE ****/
{
	/* insert local variable here */
   ARRAY_REF( slowMABuffer );
   ARRAY_REF( fastMABuffer );
   ENUM_DECLARATION(RetCode) retCode;
   int tempInteger;
   VALUE_HANDLE_INT(outBegIdx1);
   VALUE_HANDLE_INT(outNbElement1);
   VALUE_HANDLE_INT(outBegIdx2);
   VALUE_HANDLE_INT(outNbElement2);
   int lookbackTotal, lookbackSignal, lookbackLargest;
   int i;
   ENUM_DECLARATION(MAType) tempMAType;

/**** START GENCODE SECTION 4 - DO NOT DELETE THIS LINE ****/
/* Generated */ 
/* Generated */ #ifndef TA_FUNC_NO_RANGE_CHECK
/* Generated */ 
/* Generated */    /* Validate the requested output range. */
/* Generated */    if( startIdx < 0 )
/* Generated */       return ENUM_VALUE(RetCode,TA_OUT_OF_RANGE_START_INDEX,OutOfRangeStartIndex);
/* Generated */    if( (endIdx < 0) || (endIdx < startIdx))
/* Generated */       return ENUM_VALUE(RetCode,TA_OUT_OF_RANGE_END_INDEX,OutOfRangeEndIndex);
/* Generated */ 
/* Generated */    #if !defined(_JAVA)
/* Generated */    if( !inReal ) return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */    #endif /* !defined(_JAVA)*/
/* Generated */    /* min/max are checked for optInFastPeriod. */
/* Generated */    if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
/* Generated */       optInFastPeriod = 12;
/* Generated */    else if( ((int)optInFastPeriod < 2) || ((int)optInFastPeriod > 100000) )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    #if !defined(_MANAGED) && !defined(_JAVA)
/* Generated */    if( (int)optInFastMAType == TA_INTEGER_DEFAULT )
/* Generated */       optInFastMAType = (TA_MAType)0;
/* Generated */    else if( ((int)optInFastMAType < 0) || ((int)optInFastMAType > 8) )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    #endif /* !defined(_MANAGED) && !defined(_JAVA)*/
/* Generated */    /* min/max are checked for optInSlowPeriod. */
/* Generated */    if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
/* Generated */       optInSlowPeriod = 26;
/* Generated */    else if( ((int)optInSlowPeriod < 2) || ((int)optInSlowPeriod > 100000) )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    #if !defined(_MANAGED) && !defined(_JAVA)
/* Generated */    if( (int)optInSlowMAType == TA_INTEGER_DEFAULT )
/* Generated */       optInSlowMAType = (TA_MAType)0;
/* Generated */    else if( ((int)optInSlowMAType < 0) || ((int)optInSlowMAType > 8) )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    #endif /* !defined(_MANAGED) && !defined(_JAVA)*/
/* Generated */    /* min/max are checked for optInSignalPeriod. */
/* Generated */    if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
/* Generated */       optInSignalPeriod = 9;
/* Generated */    else if( ((int)optInSignalPeriod < 1) || ((int)optInSignalPeriod > 100000) )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    #if !defined(_MANAGED) && !defined(_JAVA)
/* Generated */    if( (int)optInSignalMAType == TA_INTEGER_DEFAULT )
/* Generated */       optInSignalMAType = (TA_MAType)0;
/* Generated */    else if( ((int)optInSignalMAType < 0) || ((int)optInSignalMAType > 8) )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    #endif /* !defined(_MANAGED) && !defined(_JAVA)*/
/* Generated */    #if !defined(_JAVA)
/* Generated */    if( !outMACD )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    if( !outMACDSignal )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    if( !outMACDHist )
/* Generated */       return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */ 
/* Generated */    #endif /* !defined(_JAVA) */
/* Generated */ #endif /* TA_FUNC_NO_RANGE_CHECK */
/* Generated */ 
/**** END GENCODE SECTION 4 - DO NOT DELETE THIS LINE ****/

   /* Insert TA function code here. */

   /* Make sure slow is really slower than
    * the fast period! if not, swap...
    */
   if( optInSlowPeriod < optInFastPeriod )
   {
       /* swap period */
       tempInteger     = optInSlowPeriod;
       optInSlowPeriod = optInFastPeriod;
       optInFastPeriod = tempInteger;
       /* swap type */
       tempMAType      = optInSlowMAType;
       optInSlowMAType = optInFastMAType;
       optInFastMAType = tempMAType;
   }

   /* Find the MA with the largest lookback */
   lookbackLargest = LOOKBACK_CALL(MA)( optInFastPeriod, optInFastMAType );
   tempInteger     = LOOKBACK_CALL(MA)( optInSlowPeriod, optInSlowMAType );
   if( tempInteger > lookbackLargest )
      lookbackLargest = tempInteger;

   /* Add the lookback needed for the signal line */
   lookbackSignal = LOOKBACK_CALL(MA)( optInSignalPeriod, optInSignalMAType ); 
   lookbackTotal  = lookbackSignal+lookbackLargest;

   /* Move up the start index if there is not
    * enough initial data.
    */
   if( startIdx < lookbackTotal )
      startIdx = lookbackTotal;

   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
      VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
      return ENUM_VALUE(RetCode,TA_SUCCESS,Success);
   }

   /* Allocate intermediate buffer for fast/slow MA. */
   tempInteger = (endIdx-startIdx)+1+lookbackSignal;
   ARRAY_ALLOC( fastMABuffer, tempInteger );
   #if !defined( _JAVA )
      if( !fastMABuffer )
      {
         VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
         VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
         return ENUM_VALUE(RetCode,TA_ALLOC_ERR,AllocErr);
      }
   #endif

   ARRAY_ALLOC( slowMABuffer, tempInteger );
   #if !defined( _JAVA )
      if( !slowMABuffer )
      { 
         VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
         VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
         ARRAY_FREE( fastMABuffer );
         return ENUM_VALUE(RetCode,TA_ALLOC_ERR,AllocErr);
      }
   #endif

   /* Calculate the slow MA. 
    *
    * Move back the startIdx to get enough data
    * for the signal period. That way, once the
    * signal calculation is done, all the output
    * will start at the requested 'startIdx'.
    */
   tempInteger = startIdx-lookbackSignal;
   retCode = FUNCTION_CALL(MA)( tempInteger, endIdx,
                                inReal, optInSlowPeriod, optInSlowMAType,
                                VALUE_HANDLE_OUT(outBegIdx1), VALUE_HANDLE_OUT(outNbElement1), 
							    slowMABuffer );

   if( retCode != ENUM_VALUE(RetCode,TA_SUCCESS,Success) )
   {
      VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
      VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
      ARRAY_FREE( fastMABuffer );
      ARRAY_FREE( slowMABuffer );
      return retCode;
   }

   /* Calculate the fast MA. */
   retCode = FUNCTION_CALL(MA)( tempInteger, endIdx,
                                inReal, optInFastPeriod, optInFastMAType,
                                VALUE_HANDLE_OUT(outBegIdx2), VALUE_HANDLE_OUT(outNbElement2),
							    fastMABuffer );

   if( retCode != ENUM_VALUE(RetCode,TA_SUCCESS,Success) )
   {
      VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
      VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
      ARRAY_FREE( fastMABuffer );
      ARRAY_FREE( slowMABuffer );
      return retCode;
   }

   /* Parano tests. Will be removed eventually. */
   if( (VALUE_HANDLE_GET(outBegIdx1) != tempInteger) || 
       (VALUE_HANDLE_GET(outBegIdx2) != tempInteger) || 
       (VALUE_HANDLE_GET(outNbElement1) != VALUE_HANDLE_GET(outNbElement2)) ||
       (VALUE_HANDLE_GET(outNbElement1) != (endIdx-startIdx)+1+lookbackSignal) )
   {
      VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
      VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
      ARRAY_FREE( fastMABuffer );
      ARRAY_FREE( slowMABuffer );
      return TA_INTERNAL_ERROR(119);
   }

   /* Calculate (fast MA) - (slow MA). */
   for( i=0; i < VALUE_HANDLE_GET(outNbElement1); i++ )
      fastMABuffer[i] = fastMABuffer[i] - slowMABuffer[i];

   /* Copy the result into the output for the caller. */
   ARRAY_MEMMOVE( outMACD, 0, fastMABuffer, lookbackSignal, (endIdx-startIdx)+1 );

   /* Calculate the signal/trigger line. */
   retCode = FUNCTION_CALL_DOUBLE(MA)( 0, VALUE_HANDLE_GET(outNbElement1)-1,
                                       fastMABuffer, optInSignalPeriod, optInSignalMAType,
                                       VALUE_HANDLE_OUT(outBegIdx2), VALUE_HANDLE_OUT(outNbElement2), outMACDSignal );

   ARRAY_FREE( fastMABuffer );
   ARRAY_FREE( slowMABuffer );

   if( retCode != ENUM_VALUE(RetCode,TA_SUCCESS,Success) )
   {
      VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
      VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
      return retCode;
   }

   /* Calculate the histogram. */
   for( i=0; i < VALUE_HANDLE_GET(outNbElement2); i++ )
      outMACDHist[i] = outMACD[i]-outMACDSignal[i];

   /* All done! Indicate the output limits and return success. */
   VALUE_HANDLE_DEREF(outBegIdx)     = startIdx;
   VALUE_HANDLE_DEREF(outNBElement)  = VALUE_HANDLE_GET(outNbElement2);

   return ENUM_VALUE(RetCode,TA_SUCCESS,Success);
}


/**** START GENCODE SECTION 5 - DO NOT DELETE THIS LINE ****/
/* Generated */ 
/* Generated */ #define  USE_SINGLE_PRECISION_INPUT
/* Generated */ #if !defined( _MANAGED ) && !defined( _JAVA )
/* Generated */    #undef   TA_PREFIX
/* Generated */    #define  TA_PREFIX(x) TA_S_##x
/* Generated */ #endif
/* Generated */ #undef   INPUT_TYPE
/* Generated */ #define  INPUT_TYPE float
/* Generated */ #if defined( _MANAGED )
/* Generated */ enum class Core::RetCode Core::MacdExt( int    startIdx,
/* Generated */                                         int    endIdx,
/* Generated */                                         cli::array<float>^ inReal,
/* Generated */                                         int           optInFastPeriod, /* From 2 to 100000 */
/* Generated */                                         MAType        optInFastMAType,
/* Generated */                                         int           optInSlowPeriod, /* From 2 to 100000 */
/* Generated */                                         MAType        optInSlowMAType,
/* Generated */                                         int           optInSignalPeriod, /* From 1 to 100000 */
/* Generated */                                         MAType        optInSignalMAType,
/* Generated */                                         [Out]int%    outBegIdx,
/* Generated */                                         [Out]int%    outNBElement,
/* Generated */                                         cli::array<double>^  outMACD,
/* Generated */                                         cli::array<double>^  outMACDSignal,
/* Generated */                                         cli::array<double>^  outMACDHist )
/* Generated */ #elif defined( _JAVA )
/* Generated */ public RetCode macdExt( int    startIdx,
/* Generated */                         int    endIdx,
/* Generated */                         float        inReal[],
/* Generated */                         int           optInFastPeriod, /* From 2 to 100000 */
/* Generated */                         MAType        optInFastMAType,
/* Generated */                         int           optInSlowPeriod, /* From 2 to 100000 */
/* Generated */                         MAType        optInSlowMAType,
/* Generated */                         int           optInSignalPeriod, /* From 1 to 100000 */
/* Generated */                         MAType        optInSignalMAType,
/* Generated */                         MInteger     outBegIdx,
/* Generated */                         MInteger     outNBElement,
/* Generated */                         double        outMACD[],
/* Generated */                         double        outMACDSignal[],
/* Generated */                         double        outMACDHist[] )
/* Generated */ #else
/* Generated */ TA_RetCode TA_S_MACDEXT( int    startIdx,
/* Generated */                          int    endIdx,
/* Generated */                          const float  inReal[],
/* Generated */                          int           optInFastPeriod, /* From 2 to 100000 */
/* Generated */                          TA_MAType     optInFastMAType,
/* Generated */                          int           optInSlowPeriod, /* From 2 to 100000 */
/* Generated */                          TA_MAType     optInSlowMAType,
/* Generated */                          int           optInSignalPeriod, /* From 1 to 100000 */
/* Generated */                          TA_MAType     optInSignalMAType,
/* Generated */                          int          *outBegIdx,
/* Generated */                          int          *outNBElement,
/* Generated */                          double        outMACD[],
/* Generated */                          double        outMACDSignal[],
/* Generated */                          double        outMACDHist[] )
/* Generated */ #endif
/* Generated */ {
/* Generated */    ARRAY_REF( slowMABuffer );
/* Generated */    ARRAY_REF( fastMABuffer );
/* Generated */    ENUM_DECLARATION(RetCode) retCode;
/* Generated */    int tempInteger;
/* Generated */    VALUE_HANDLE_INT(outBegIdx1);
/* Generated */    VALUE_HANDLE_INT(outNbElement1);
/* Generated */    VALUE_HANDLE_INT(outBegIdx2);
/* Generated */    VALUE_HANDLE_INT(outNbElement2);
/* Generated */    int lookbackTotal, lookbackSignal, lookbackLargest;
/* Generated */    int i;
/* Generated */    ENUM_DECLARATION(MAType) tempMAType;
/* Generated */  #ifndef TA_FUNC_NO_RANGE_CHECK
/* Generated */     if( startIdx < 0 )
/* Generated */        return ENUM_VALUE(RetCode,TA_OUT_OF_RANGE_START_INDEX,OutOfRangeStartIndex);
/* Generated */     if( (endIdx < 0) || (endIdx < startIdx))
/* Generated */        return ENUM_VALUE(RetCode,TA_OUT_OF_RANGE_END_INDEX,OutOfRangeEndIndex);
/* Generated */     #if !defined(_JAVA)
/* Generated */     if( !inReal ) return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #endif 
/* Generated */     if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
/* Generated */        optInFastPeriod = 12;
/* Generated */     else if( ((int)optInFastPeriod < 2) || ((int)optInFastPeriod > 100000) )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #if !defined(_MANAGED) && !defined(_JAVA)
/* Generated */     if( (int)optInFastMAType == TA_INTEGER_DEFAULT )
/* Generated */        optInFastMAType = (TA_MAType)0;
/* Generated */     else if( ((int)optInFastMAType < 0) || ((int)optInFastMAType > 8) )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #endif 
/* Generated */     if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
/* Generated */        optInSlowPeriod = 26;
/* Generated */     else if( ((int)optInSlowPeriod < 2) || ((int)optInSlowPeriod > 100000) )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #if !defined(_MANAGED) && !defined(_JAVA)
/* Generated */     if( (int)optInSlowMAType == TA_INTEGER_DEFAULT )
/* Generated */        optInSlowMAType = (TA_MAType)0;
/* Generated */     else if( ((int)optInSlowMAType < 0) || ((int)optInSlowMAType > 8) )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #endif 
/* Generated */     if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
/* Generated */        optInSignalPeriod = 9;
/* Generated */     else if( ((int)optInSignalPeriod < 1) || ((int)optInSignalPeriod > 100000) )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #if !defined(_MANAGED) && !defined(_JAVA)
/* Generated */     if( (int)optInSignalMAType == TA_INTEGER_DEFAULT )
/* Generated */        optInSignalMAType = (TA_MAType)0;
/* Generated */     else if( ((int)optInSignalMAType < 0) || ((int)optInSignalMAType > 8) )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #endif 
/* Generated */     #if !defined(_JAVA)
/* Generated */     if( !outMACD )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     if( !outMACDSignal )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     if( !outMACDHist )
/* Generated */        return ENUM_VALUE(RetCode,TA_BAD_PARAM,BadParam);
/* Generated */     #endif 
/* Generated */  #endif 
/* Generated */    if( optInSlowPeriod < optInFastPeriod )
/* Generated */    {
/* Generated */        tempInteger     = optInSlowPeriod;
/* Generated */        optInSlowPeriod = optInFastPeriod;
/* Generated */        optInFastPeriod = tempInteger;
/* Generated */        tempMAType      = optInSlowMAType;
/* Generated */        optInSlowMAType = optInFastMAType;
/* Generated */        optInFastMAType = tempMAType;
/* Generated */    }
/* Generated */    lookbackLargest = LOOKBACK_CALL(MA)( optInFastPeriod, optInFastMAType );
/* Generated */    tempInteger     = LOOKBACK_CALL(MA)( optInSlowPeriod, optInSlowMAType );
/* Generated */    if( tempInteger > lookbackLargest )
/* Generated */       lookbackLargest = tempInteger;
/* Generated */    lookbackSignal = LOOKBACK_CALL(MA)( optInSignalPeriod, optInSignalMAType ); 
/* Generated */    lookbackTotal  = lookbackSignal+lookbackLargest;
/* Generated */    if( startIdx < lookbackTotal )
/* Generated */       startIdx = lookbackTotal;
/* Generated */    if( startIdx > endIdx )
/* Generated */    {
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
/* Generated */       return ENUM_VALUE(RetCode,TA_SUCCESS,Success);
/* Generated */    }
/* Generated */    tempInteger = (endIdx-startIdx)+1+lookbackSignal;
/* Generated */    ARRAY_ALLOC( fastMABuffer, tempInteger );
/* Generated */    #if !defined( _JAVA )
/* Generated */       if( !fastMABuffer )
/* Generated */       {
/* Generated */          VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
/* Generated */          VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
/* Generated */          return ENUM_VALUE(RetCode,TA_ALLOC_ERR,AllocErr);
/* Generated */       }
/* Generated */    #endif
/* Generated */    ARRAY_ALLOC( slowMABuffer, tempInteger );
/* Generated */    #if !defined( _JAVA )
/* Generated */       if( !slowMABuffer )
/* Generated */       { 
/* Generated */          VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
/* Generated */          VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
/* Generated */          ARRAY_FREE( fastMABuffer );
/* Generated */          return ENUM_VALUE(RetCode,TA_ALLOC_ERR,AllocErr);
/* Generated */       }
/* Generated */    #endif
/* Generated */    tempInteger = startIdx-lookbackSignal;
/* Generated */    retCode = FUNCTION_CALL(MA)( tempInteger, endIdx,
/* Generated */                                 inReal, optInSlowPeriod, optInSlowMAType,
/* Generated */                                 VALUE_HANDLE_OUT(outBegIdx1), VALUE_HANDLE_OUT(outNbElement1), 
/* Generated */ 							    slowMABuffer );
/* Generated */    if( retCode != ENUM_VALUE(RetCode,TA_SUCCESS,Success) )
/* Generated */    {
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
/* Generated */       ARRAY_FREE( fastMABuffer );
/* Generated */       ARRAY_FREE( slowMABuffer );
/* Generated */       return retCode;
/* Generated */    }
/* Generated */    retCode = FUNCTION_CALL(MA)( tempInteger, endIdx,
/* Generated */                                 inReal, optInFastPeriod, optInFastMAType,
/* Generated */                                 VALUE_HANDLE_OUT(outBegIdx2), VALUE_HANDLE_OUT(outNbElement2),
/* Generated */ 							    fastMABuffer );
/* Generated */    if( retCode != ENUM_VALUE(RetCode,TA_SUCCESS,Success) )
/* Generated */    {
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
/* Generated */       ARRAY_FREE( fastMABuffer );
/* Generated */       ARRAY_FREE( slowMABuffer );
/* Generated */       return retCode;
/* Generated */    }
/* Generated */    if( (VALUE_HANDLE_GET(outBegIdx1) != tempInteger) || 
/* Generated */        (VALUE_HANDLE_GET(outBegIdx2) != tempInteger) || 
/* Generated */        (VALUE_HANDLE_GET(outNbElement1) != VALUE_HANDLE_GET(outNbElement2)) ||
/* Generated */        (VALUE_HANDLE_GET(outNbElement1) != (endIdx-startIdx)+1+lookbackSignal) )
/* Generated */    {
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
/* Generated */       ARRAY_FREE( fastMABuffer );
/* Generated */       ARRAY_FREE( slowMABuffer );
/* Generated */       return TA_INTERNAL_ERROR(119);
/* Generated */    }
/* Generated */    for( i=0; i < VALUE_HANDLE_GET(outNbElement1); i++ )
/* Generated */       fastMABuffer[i] = fastMABuffer[i] - slowMABuffer[i];
/* Generated */    ARRAY_MEMMOVE( outMACD, 0, fastMABuffer, lookbackSignal, (endIdx-startIdx)+1 );
/* Generated */    retCode = FUNCTION_CALL_DOUBLE(MA)( 0, VALUE_HANDLE_GET(outNbElement1)-1,
/* Generated */                                        fastMABuffer, optInSignalPeriod, optInSignalMAType,
/* Generated */                                        VALUE_HANDLE_OUT(outBegIdx2), VALUE_HANDLE_OUT(outNbElement2), outMACDSignal );
/* Generated */    ARRAY_FREE( fastMABuffer );
/* Generated */    ARRAY_FREE( slowMABuffer );
/* Generated */    if( retCode != ENUM_VALUE(RetCode,TA_SUCCESS,Success) )
/* Generated */    {
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outBegIdx);
/* Generated */       VALUE_HANDLE_DEREF_TO_ZERO(outNBElement);
/* Generated */       return retCode;
/* Generated */    }
/* Generated */    for( i=0; i < VALUE_HANDLE_GET(outNbElement2); i++ )
/* Generated */       outMACDHist[i] = outMACD[i]-outMACDSignal[i];
/* Generated */    VALUE_HANDLE_DEREF(outBegIdx)     = startIdx;
/* Generated */    VALUE_HANDLE_DEREF(outNBElement)  = VALUE_HANDLE_GET(outNbElement2);
/* Generated */    return ENUM_VALUE(RetCode,TA_SUCCESS,Success);
/* Generated */ }
/* Generated */ 
/* Generated */ #if defined( _MANAGED )
/* Generated */ }}} // Close namespace TicTacTec.TA.Lib
/* Generated */ #endif
/**** END GENCODE SECTION 5 - DO NOT DELETE THIS LINE ****/

