//
// Created by vivi on 07/02/2018.
//

#pragma once

#include <cstdio>
#include <cstdlib>

/*
 * Safe Delete
 */
#define HairEngine_SafeDelete(item_) if(item_ != nullptr) delete item_
#define HairEngine_SafeDeleteArray(item_) if(item_ != nullptr) delete [] item_

/*
 * Some useful definitions for debugging so that we could access all the private data in the class
 * */
#if defined(NDEBUG) && !defined(HAIRENGINE_TEST)
	#define HairEngine_Public public
	#define HairEngine_Protected protected
	#define HairEngine_Private private
#else
	#define HairEngine_Public public
	#define HairEngine_Protected public
	#define HairEngine_Private public
#endif

/*
 * Debug
 */
#ifdef NDEBUG
#define HairEngine_DebugIf if(0)
#else
#define HAIRENGINE_DEBUG_ON
#define HairEngine_DebugIf If(1)
#endif

/*
* Assertion
*/
#ifdef NDEBUG
#define HairEngine_Assert(expr_) ((void)0)
#else
#define HairEngine_Assert(expr_) if(!(expr_))Error("Assertion \"%s\" failed in \"%s\", line %d.", #expr_, __FILE__, __LINE__)
#endif
