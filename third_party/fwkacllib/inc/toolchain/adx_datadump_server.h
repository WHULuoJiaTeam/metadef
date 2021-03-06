/**
* @file adx_datadump_server.h
*
* Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
* Copyright 2021, 2022 Huawei Technologies Co., Ltd
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef ADX_DATADUMP_SERVER_H
#define ADX_DATADUMP_SERVER_H
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief initialize server for normal datadump function.
 * @return
 *      IDE_DAEMON_OK:    datadump server init success
 *      IDE_DAEMON_ERROR: datadump server init failed
 */
int AdxDataDumpServerInit();

/**
 * @brief uninitialize server for normal datadump function.
 * @return
 *      IDE_DAEMON_OK:    datadump server uninit success
 *      IDE_DAEMON_ERROR: datadump server uninit failed
 */
int AdxDataDumpServerUnInit();

#ifdef __cplusplus
}
#endif
#endif

